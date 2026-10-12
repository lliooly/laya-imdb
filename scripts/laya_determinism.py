"""Explicit diagnostic protocols; never enabled by the formal trainer."""
import json
import os


def configure_runtime(torch, profile):
    if profile not in ('baseline', 'no-checkpoint', 'math-sdp', 'strict'):
        raise ValueError('Unknown runtime profile: ' + profile)
    if profile == 'strict':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before importing Torch')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if profile == 'math-sdp':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def compare_workers(left, right):
    a = [json.loads(line) for line in (left / 'trace/updates.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (right / 'trace/updates.jsonl').read_text().splitlines()]
    if len(a) != 16 or len(b) != 16:
        raise ValueError('Expected exactly 16 updates in both workers')
    differences = [{'update': x['update'], 'fields': [key for key in sorted(x.keys() | y.keys())
                    if x.get(key) != y.get(key)]} for x, y in zip(a, b) if x != y]
    gradients_a = json.loads((left / 'trace/first-gradients.json').read_text())
    gradients_b = json.loads((right / 'trace/first-gradients.json').read_text())
    if gradients_a.keys() != gradients_b.keys():
        raise ValueError('Parameter names differ between runs')
    changed = [name for name in gradients_a if gradients_a[name] != gradients_b[name]]
    modules = {}
    for name in changed:
        module = name.rsplit('.', 1)[0]
        modules[module] = modules.get(module, 0) + 1
    environment_a = json.loads((left / 'environment.json').read_text())
    environment_b = json.loads((right / 'environment.json').read_text())
    excluded = {'entry_sha256', 'executed_entry'}
    environment_diff = [key for key in sorted(environment_a.keys() | environment_b.keys())
                        if key not in excluded and environment_a.get(key) != environment_b.get(key)]
    return {'identical': not differences and not changed and not environment_diff,
            'compared_updates': len(a), 'differences': differences,
            'first_gradient_changed_parameters': changed,
            'first_gradient_changed_modules': modules,
            'parameter_count': len(gradients_a),
            'first_parameter_in_model_order': changed[0] if changed else None,
            'parameter_order_is_not_backward_execution_order': True,
            'environment_difference_fields': environment_diff}
