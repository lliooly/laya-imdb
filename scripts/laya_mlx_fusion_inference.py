"""MLX FP16 execution of the frozen fusion; shared, byte-identical CPU input preparation."""
import json
import time
import statistics
import numpy as np
from laya_fusion_inference import FrozenFusionPredictor
from export_laya_posthoc import input_digest

class MLXFusionPredictor(FrozenFusionPredictor):
    def __init__(self, root, checkpoint_root, assets, frozen_fit, device='mlx', batch_size=8):
        if device != 'mlx':
            raise ValueError('MLX GPU only')
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError('Metal GPU is required')
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(2 * 1024**3)
        self.mx = mx
        # PyTorch supplies only the pinned tokenizer/CPU sequence builder, never a model forward.
        super().__init__(root, checkpoint_root, assets, frozen_fit, 'cpu', batch_size)

    def load(self, name):
        if name in self.models:
            return
        from laya_mlx import Agent
        from transformers import AutoTokenizer
        from mlx.utils import tree_flatten
        started = time.perf_counter()
        cfg = json.loads((self.folder(name)/'rl_agent_config.json').read_text())
        if ((cfg['max_len'], cfg['head_max_len']) != (512, 192)
                or self.upstream.uses_parallel_layout(cfg) or cfg.get('temperature_by_options')
                or not np.isfinite(cfg['temperature'][0]) or cfg['temperature'][0] <= 0):
            raise ValueError('Unexpected layout or native temperature')
        agent = Agent(str(self.folder(name)), device='gpu', dtype='float16',
                      compile=False, cache_prompts=False, batch_size=self.batch_size)
        for key, value in tree_flatten(agent.model.parameters()):
            if value.dtype != self.mx.float16 or not bool(self.mx.all(self.mx.isfinite(value)).item()):
                raise FloatingPointError('Invalid FP16 parameter: '+key)
        self.models[name] = agent
        self.tokenizers[name] = AutoTokenizer.from_pretrained(str(self.folder(name)/'tokenizer'), local_files_only=True)
        self.configs[name] = cfg
        self.mx.synchronize()
        self.load_seconds[name] = time.perf_counter()-started

    def _infer(self, name, items, indices, stats, ids, reuse=None):
        from laya_mlx.agent import collate_items
        tok, agent, cfg = self.tokenizers[name], self.models[name], self.configs[name]
        result = [dict(r, reused_head_prediction=True) for r in reuse] if reuse is not None else [None]*len(items)
        for start in range(0, len(indices), self.batch_size):
            ii = indices[start:start+self.batch_size]
            prepared = [self.upstream.encode_item(tok, items[i], 512, 192, [0, 1], parallel=False) for i in ii]
            batch = collate_items(prepared, tok.pad_token_id)
            logits, actions = agent.forward(batch)
            z = np.asarray(logits, dtype=np.float64)
            if z.shape != (len(ii), 2) or not np.isfinite(z).all() or not np.isfinite(np.asarray(actions)).all():
                raise FloatingPointError('Nonfinite MLX outputs; no dtype fallback')
            for j, i in enumerate(ii):
                p = np.exp((z[j]-z[j].max())/float(cfg['temperature'][0]))
                p /= p.sum()
                result[i] = {'sample_id': ids[i], 'input_sha256': input_digest(prepared[j]),
                    'logits': z[j].tolist(), 'probabilities': p.tolist(), 'pred_label': int(z[j].argmax()),
                    **stats[i], 'reused_head_prediction': False}
        return result


def benchmark(predictor, texts, mode, spec):
    mx = predictor.mx
    mx.clear_cache()
    for _ in range(spec['warmup']):
        predictor.predict(texts, mode=mode)
        mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    durations, forwards = [], []
    for _ in range(spec['repetitions']):
        mx.synchronize()
        started = time.perf_counter()
        result = predictor.predict(texts, mode=mode)
        mx.synchronize()
        durations.append(time.perf_counter()-started)
        forwards.append(result['forward_rows'])
    median = statistics.median(durations)
    return {'mode': mode, 'review_count': len(texts), 'seconds': durations, 'median_seconds': median,
        'reviews_per_second': len(texts)/median, 'amortized_milliseconds_per_review': median/len(texts)*1000,
        'forward_rows_per_repetition': forwards, 'truncated_count': sum(s['truncated'] for s in result['stats']),
        'review_token_counts': [s['review_tokens'] for s in result['stats']],
        'memory': {'peak_active_bytes': mx.get_peak_memory(), 'active_bytes': mx.get_active_memory(),
                   'cache_bytes': mx.get_cache_memory(),
                   'method': 'MLX allocator peak including resident weights; excludes CPU/PyTorch tokenizer overhead and total process RSS'},
        'scope': 'tokenization, preparation, transfers, forwards and fusion; excludes model loading',
        'is_single_request_latency': False}
