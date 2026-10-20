"""Verify installed distributions against both original locks and exercise CUDA."""
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CODE = r'''
import importlib.metadata as md,json,sys,sysconfig
import torch,transformers,numpy,safetensors
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.is_bf16_supported(including_emulation=False)
x=torch.randn(32,32,device='cuda',requires_grad=True)
with torch.autocast('cuda',dtype=torch.bfloat16):
 y=x@x
loss=y.float().square().mean();loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print(json.dumps(dict(python=sys.version,packages={p.metadata['Name'].lower().replace('_','-'):p.version for p in md.distributions()},gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,bf16_finite_backward=True,python_include=sysconfig.get_paths()['include'])))
'''


def main():
    results = {}
    for name in ('laya-cu126', 'deberta-cu126'):
        folder = ROOT/'environments'/name
        lock = tomllib.loads((folder/'uv.lock').read_text())
        expected = {p['name']: p['version'] for p in lock['package'] if 'virtual' not in p['source']}
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
        value = json.loads(subprocess.check_output([str(folder/'.venv/bin/python'), '-c', CODE], env=env, text=True))
        differences = {p: [version, value['packages'].get(p)] for p, version in expected.items()
                       if value['packages'].get(p) != version}
        if differences:
            raise ValueError('Installed distributions differ from original lock: '+str(differences))
        value['verified_locked_packages'] = len(expected)
        results[name] = value
    if results['laya-cu126']['python'] != results['deberta-cu126']['python']:
        raise ValueError('Python patch versions differ')
    out = ROOT/'reports/paper-gpu-supplement-20261019-v1/environment-admission.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(dict(state='passed', environments=results, test_accessed=False,
        mirror='Tsinghua PyPI mirror; original hashes required; torch official URL preserved'), indent=2)+'\n')
    print('LOCKED_ENVIRONMENTS_AND_CUDA_PASSED', flush=True)


if __name__ == '__main__':
    main()
