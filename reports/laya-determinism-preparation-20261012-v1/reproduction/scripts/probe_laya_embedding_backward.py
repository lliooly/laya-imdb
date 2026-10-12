"""Synthetic ModernBERT embedding reproduction; reads configuration, never weights/data."""
import argparse
import os
from pathlib import Path

from common import ROOT, write_json, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['compiled', 'eager', 'compiled-strict'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / 'reports') or out.exists():
        raise ValueError('Use a new report path')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    if args.profile == 'compiled-strict':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import ModernBertEmbeddings
    from laya_short_trace import tensor_hash
    from laya_determinism import configure_runtime
    if args.profile == 'compiled-strict':
        configure_runtime(torch, 'strict')
    assert torch.cuda.is_available()
    config_path = ROOT / '.cache/laya/english/encoder/config.json'
    config = ModernBertConfig.from_pretrained(str(config_path.parent), local_files_only=True)
    config.reference_compile = args.profile != 'eager'
    torch.manual_seed(43)
    model = ModernBertEmbeddings(config).to('cuda').train()
    ids = torch.randint(0, 1024, (8, 512), device='cuda')
    cotangent = torch.randn((8, 512, config.hidden_size), device='cuda')
    rng = torch.cuda.get_rng_state()
    parameters = dict(model.named_parameters())
    result = {'profile': args.profile, 'synthetic_inputs': True, 'real_training_inputs': False,
              'data_accessed': False, 'model_weights_accessed': False, 'test_accessed': False,
              'configuration_sha256': file_digest(config_path),
              'shape': list(model.tok_embeddings.weight.shape), 'input_shape': list(ids.shape),
              'unique_tokens': int(ids.unique().numel()), 'embedding_dropout': config.embedding_dropout,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'rows': []}
    first = None
    try:
        for repeat in range(4):
            torch.cuda.set_rng_state(rng)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                output = model(ids)
                gradients = torch.autograd.grad(output, tuple(parameters.values()), grad_outputs=cotangent)
            values = {name: grad.detach().cpu() for name, grad in zip(parameters, gradients)}
            assert all(torch.isfinite(value).all() for value in values.values())
            if first is None:
                first = values
            result['rows'].append({'repeat': repeat + 1, 'output_sha256': tensor_hash(output),
                'rng_after_sha256': tensor_hash(torch.cuda.get_rng_state()),
                'gradient_sha256': {name:tensor_hash(value) for name,value in values.items()},
                'gradient_differences_vs_first': {name:{'different_elements':int((first[name] != value).sum()),
                    'max_absolute_difference':float((first[name] - value).abs().max())} for name,value in values.items()},
                'embedding_operators':{event.key:event.count for event in profiler.key_averages()
                                       if 'embedding' in event.key.lower() or 'compiled' in event.key.lower()}})
        rows = result['rows']
        result.update(state='completed', outputs_identical=len({r['output_sha256'] for r in rows}) == 1,
                      rng_identical=len({r['rng_after_sha256'] for r in rows}) == 1,
                      gradients_identical=all(r['gradient_sha256'] == rows[0]['gradient_sha256'] for r in rows))
    except BaseException as error:
        result.update(state='failed', error=repr(error))
        raise
    finally:
        write_json(out, result)


if __name__ == '__main__':
    main()
