"""Fresh-process original/control diagnostics using the full historical budget."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from common import ROOT, file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--worker', choices=['original', 'control'])
    parser.add_argument('--config')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1800:
        raise ValueError('Per-worker time limit must be within 1..1800 seconds')
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Output must be under repository reports')
    historical = ROOT / 'reports/laya-resampling45-control-drift-audit-v2/historical-train_laya.reconstructed.py'
    assert file_digest(historical) == 'e0310de3c0c9dd21e14b0a64c03899790d4b498bdd2c8f98374594ba753fb581'
    current = ROOT / 'reports/laya-resampling-45-preparation-v2/reproduction/scripts/train_laya.py'
    assert file_digest(current) == '18a6bad97994dba2ef4e265d703455ba6bf4254c6add70822cee948264b0b690'
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            os.environ[key] = '1'
        import torch
        import laya.train as upstream
        from laya_short_trace import DiagnosticStop, short_trace
        entry = historical if args.worker == 'original' else current
        spec = json.loads(Path(args.config).read_text())
        formal = json.loads((ROOT / 'reports/laya-rtx4090-ce-resampling-control-pool45-s43-epochs2-fp32-v2/manifest.json').read_text())
        for name, field in [('cuda_numerics', 'cuda_guard_sha256'),
                            ('laya_optimization', 'optimization_adapter_sha256'),
                            ('laya_epoch_checkpoints', 'epoch_checkpoint_adapter_sha256'),
                            ('laya_resampling', 'sampling_adapter_sha256')]:
            if file_digest(ROOT / 'scripts' / f'{name}.py') != formal[field]:
                raise ValueError('Training dependency changed since formal experiment: ' + name)
        assert spec['training'] == formal['spec']['training']
        assert spec['optimization'] == formal['spec']['optimization']
        assert spec['training']['grad_accum'] == 1 and spec['training']['amp'] is False
        assert spec['train_limit'] is None and spec['training']['epochs'] == 2
        environment = {
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'mem_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'entry_sha256': file_digest(entry), 'executed_entry': str(entry.relative_to(ROOT)),
            'trace_module_sha256': file_digest(ROOT / 'scripts/laya_short_trace.py'),
            'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'instrumented_run_is_not_uninstrumented_historical_reproduction': True,
        }
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        environment['nvidia_smi'] = {'exit_code': result.returncode, 'output': result.stdout, 'error': result.stderr}
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'environment.json', environment)
        sys.argv = [str(entry), '--config', str(Path(args.config).resolve())]
        try:
            with short_trace(torch, upstream, out / 'trace'):
                runpy.run_path(str(entry), run_name='__main__')
        except DiagnosticStop:
            optimization = json.loads((ROOT / 'reports' / spec['run_id'] / 'optimization_diagnostics.json').read_text())
            assert optimization['completed_optimizer_updates'] == 16
            write_json(out / 'result.json', {'state': 'diagnostic_completed', 'completed_updates': 16,
                                           'full_schedule_updates': optimization['total_optimizer_updates'],
                                           'expected_training_entry_status': 'failed_due_to_intentional_DiagnosticStop',
                                           'test_accessed': False})
        return
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((ROOT / 'configs/laya-rtx4090-ce-resampling-control-pool45-s43-epochs2-fp32-v2.json').read_text())
    status = {'state': 'running', 'completed': [], 'test_accessed': False}
    write_json(out / 'status.json', status)
    try:
        for mode in ('original', 'control'):
            for repeat in (1, 2):
                label = f'{mode}-{repeat}'
                config = json.loads(json.dumps(spec))
                config['run_id'] = f'{out.name}-{label}'
                if mode == 'original':
                    config.pop('sampling')
                path = out / f'{label}-config.json'
                write_json(path, config)
                with (out / f'{label}.log').open('x') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                                    '--config', str(path), '--out', str(out / label)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=args.max_seconds, check=True)
                status['completed'].append(label)
                write_json(out / 'status.json', status)
        records = {label: [json.loads(line) for line in (out / label / 'trace/updates.jsonl').read_text().splitlines()]
                   for label in status['completed']}
        comparisons = {}
        for a, b in [('original-1', 'original-2'), ('control-1', 'control-2'), ('original-1', 'control-1')]:
            assert len(records[a]) == len(records[b]) == 16
            differences = [{'update': x['update'], 'fields': [key for key in sorted(x.keys() | y.keys()) if x.get(key) != y.get(key)]}
                           for x, y in zip(records[a], records[b]) if x != y]
            env_a = json.loads((out / a / 'environment.json').read_text())
            env_b = json.loads((out / b / 'environment.json').read_text())
            environment_differences = [key for key in sorted(env_a.keys() | env_b.keys())
                                       if key not in ('entry_sha256', 'executed_entry') and env_a.get(key) != env_b.get(key)]
            comparisons[f'{a}_vs_{b}'] = {'identical': not differences, 'differences': differences,
                                          'environment_difference_fields': environment_differences}
        write_json(out / 'comparison.json', comparisons)
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        write_json(out / 'status.json', status)


if __name__ == '__main__':
    main()
