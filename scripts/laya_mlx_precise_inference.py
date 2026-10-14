"""Explicit full-FP32 Metal inference with reduced-precision matrix operations disabled."""
import os
import json
import time
import numpy as np
from laya_mlx_fusion_inference import MLXFusionPredictor

class PreciseMLXFusionPredictor(MLXFusionPredictor):
    def __init__(self, root, checkpoint_root, assets, frozen_fit, device='mlx-fp32-precise', batch_size=8):
        if device != 'mlx-fp32-precise' or os.environ.get('MLX_ENABLE_TF32') != '0':
            raise RuntimeError('Launch a fresh process with MLX_ENABLE_TF32=0 for precise MLX FP32')
        super().__init__(root, checkpoint_root, assets, frozen_fit, 'mlx', batch_size)

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
        agent = Agent(str(self.folder(name)), device='gpu', dtype='float32',
                      compile=False, cache_prompts=False, batch_size=self.batch_size)
        for key, value in tree_flatten(agent.model.parameters()):
            if value.dtype != self.mx.float32 or not bool(self.mx.all(self.mx.isfinite(value)).item()):
                raise FloatingPointError('Invalid FP32 parameter: '+key)
        self.models[name] = agent
        self.tokenizers[name] = AutoTokenizer.from_pretrained(str(self.folder(name)/'tokenizer'), local_files_only=True)
        self.configs[name] = cfg
        self.mx.synchronize()
        self.load_seconds[name] = time.perf_counter()-started

