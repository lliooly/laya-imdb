"""Opt-in formal CUDA determinism; configure workspace before importing Laya/Torch."""
import os
import sys


def prepare_runtime(spec):
    profile = spec.get('runtime_profile')
    if profile not in (None, 'strict'):
        raise ValueError('Unsupported formal runtime profile')
    if profile is None:
        return None
    if not spec['device'].startswith('cuda') or spec['training']['amp']:
        raise ValueError('Strict protocol requires CUDA FP32 without AMP')
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    torch = sys.modules.get('torch')
    if previous != ':4096:8' and torch is not None and torch.cuda.is_initialized():
        raise RuntimeError('Cannot change CUBLAS workspace after CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return profile


def apply_runtime(torch, profile):
    if profile is None:
        return None
    if profile != 'strict' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Strict workspace was not prepared')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {'profile': profile,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}
