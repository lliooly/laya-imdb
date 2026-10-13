"""Fixed-weight MLX sensitivity probe on AI-drafted seen-training triplets only."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import time

from common import ROOT, file_digest, write_json
from semantic_pair_pilot import VARIANTS, digest, load_source


def summarize(rows):
    correct = {name: sum(r['predictions'][name]['label'] == r['label'] for r in rows)
               for name in VARIANTS}
    transitions = {}
    for name in VARIANTS[1:]:
        fixed, regressed, changed = [], [], []
        for row in rows:
            a, b = row['predictions']['original']['label'], row['predictions'][name]['label']
            if a != b:
                changed.append(row['parent_id'])
            if a != row['label'] and b == row['label']:
                fixed.append(row['parent_id'])
            if a == row['label'] and b != row['label']:
                regressed.append(row['parent_id'])
        transitions[name] = {'fixed': fixed, 'regressed': regressed, 'changed': changed,
                             'pair_both_correct': sum(r['predictions']['original']['label'] == r['label']
                                 and r['predictions'][name]['label'] == r['label'] for r in rows)}
    return {'count': len(rows), 'correct_against_unverified_inherited_labels': correct,
            'transitions': transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/laya-mps-ce-epochs2-l512-s42-fp32-v1')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/laya-l512-s42-ce/manifest.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use an unused probe output')
    pack_sha = file_digest(args.pack)
    pack = json.loads(args.pack.read_text())
    source, gradient, manifest, source_sha = load_source(args.manifest)
    if pack['source_sha256'] != source_sha or pack['manifest_sha256'] != file_digest(args.manifest):
        raise ValueError('Pack source provenance mismatch')
    if pack['question'] != manifest['question']:
        raise ValueError('Question differs from frozen task')
    parents = [r['parent_id'] for r in pack['rows']]
    if not parents or len(parents) != len(set(parents)) or not set(parents) <= set(gradient):
        raise ValueError('Expected unique gradient-training parents')
    for row in pack['rows']:
        if row['label'] != source[row['parent_id']]['label'] or row['variants']['original']['text'] != source[row['parent_id']]['text']:
            raise ValueError('Original source text or label mismatch')
        if set(row['variants']) != set(VARIANTS):
            raise ValueError('Missing triplet variant')
        if any(digest(v['text']) != v['sha256'] for v in row['variants'].values()):
            raise ValueError('Variant hash mismatch')
    import os
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    import mlx.core as mx
    import numpy as np
    from laya_mlx import Agent
    from laya_mlx.agent import collate_items
    weights = args.checkpoint / 'model.safetensors'
    weight_sha = file_digest(weights)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    write_json(args.out / 'status.json', {'state': 'running', 'training_performed': False})
    try:
        mx.set_cache_limit(1024**3)
        agent = Agent(str(args.checkpoint), device='gpu', dtype='float16', compile=False, cache_prompts=False)
        if (agent.cfg['max_len'], agent.cfg['head_max_len']) != (512, 192):
            raise ValueError('Unexpected checkpoint input budget')
        # Check the saved tokenizer has the same semantics as the frozen source.
        lock = json.loads((ROOT / 'configs/laya_model_lock.json').read_text())
        tokenizer_path = ROOT / lock['local_dir'] / 'tokenizer/tokenizer.json'
        if file_digest(tokenizer_path) != pack['tokenizer_sha256']:
            raise ValueError('Frozen tokenizer changed')
        def semantics(path):
            value = json.loads(path.read_text())
            value.pop('padding', None); value.pop('truncation', None)
            return value
        if semantics(tokenizer_path) != semantics(args.checkpoint / 'tokenizer/tokenizer.json'):
            raise ValueError('Checkpoint tokenizer differs')
        results = []
        with (args.out / 'predictions.jsonl').open('w') as stream:
            for index, row in enumerate(pack['rows'], 1):
                predictions = {}
                for name in VARIANTS:
                    value = row['variants'][name]
                    items, _ = agent.prepare(value['text'], {'sentiment': pack['question']})
                    if len(items) != 1 or len(items[0]['markers']) != 2:
                        raise ValueError('Unexpected question encoding')
                    batch = collate_items(items, agent.tok.pad_token_id)
                    z = np.asarray(agent.forward(batch)[0], dtype=np.float64)[0]
                    if z.shape != (2,) or not np.isfinite(z).all():
                        raise FloatingPointError('Expected two finite logits')
                    predictions[name] = {'label': int(z.argmax()), 'logits': z.tolist(),
                                         'text_sha256': value['sha256'],
                                         'input_sha256': digest(json.dumps({k: items[0][k] for k in ('ids', 'markers', 'qtype')}, sort_keys=True)),
                                         'input_tokens': len(items[0]['ids'])}
                result = {k: row[k] for k in ('parent_id', 'label', 'category')}
                result['predictions'] = predictions
                results.append(result)
                stream.write(json.dumps(result) + '\n'); stream.flush()
                print(f'{index}/{len(pack["rows"])} triplets', flush=True)
        if file_digest(weights) != weight_sha or file_digest(args.pack) != pack_sha:
            raise ValueError('Checkpoint or draft pack changed during probe')
        summary = {'state': 'completed_fixed_weight_diagnostic', 'training_performed': False,
                   'test_accessed': False, 'validation_body_accessed': False,
                   'seen_training_sources': True, 'human_review_completed': False,
                   'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': weight_sha,
                   'pack_sha256': pack_sha, 'script_sha256': file_digest(Path(__file__)),
                   'dtype': 'float16', 'backend': 'mlx', 'compile': False,
                   'versions': {k: version(k) for k in ('mlx', 'laya-mlx')},
                   'elapsed_seconds': time.monotonic() - started,
                   'overall': summarize(results),
                   'by_category': {c: summarize([r for r in results if r['category'] == c])
                                   for c in sorted({r['category'] for r in results})},
                   'interpretation': 'Expression sensitivity on seen training sources with AI-drafted variants; not evidence of training gain or generalization.'}
        write_json(args.out / 'summary.json', summary)
        write_json(args.out / 'status.json', {'state': 'completed', 'training_performed': False})
        print(json.dumps(summary['overall'], ensure_ascii=False))
    except BaseException as exc:
        write_json(args.out / 'status.json', {'state': 'failed', 'error': str(exc), 'training_performed': False})
        raise


if __name__ == '__main__':
    main()
