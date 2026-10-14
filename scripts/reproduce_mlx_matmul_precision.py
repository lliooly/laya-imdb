"""Model/data-free deterministic FP32 matmul reproduction, run in a fresh process."""
import argparse
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    import mlx.core as mx
    if not mx.metal.is_available():raise RuntimeError('Metal GPU required')
    rng=np.random.default_rng(20261014)
    x=rng.standard_normal((8,512,1024),dtype=np.float32)*np.float32(.5)
    w=rng.standard_normal((3072,1024),dtype=np.float32)*np.float32(.03)
    outputs={}
    for name,dev in [('cpu',mx.cpu),('gpu',mx.gpu)]:
        with mx.stream(dev):
            a=mx.array(x);b=mx.array(w);y=a@b.T;mx.eval(y)
            outputs[name]=np.asarray(y).copy()
    delta=np.abs(outputs['gpu']-outputs['cpu'])
    result={'mlx_version':version('mlx'),'hardware':mx.device_info(),
        'mlx_enable_tf32':os.environ.get('MLX_ENABLE_TF32','unset'),
        'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'weight_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
        'input_shape':list(x.shape),'weight_shape':list(w.shape),'dtype':'float32',
        'max_abs_delta':float(delta.max()),'rms_delta':float(np.sqrt(np.mean(delta**2))),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_or_dataset_loaded':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
