"""Audited one-expression adaptation of the pinned epoch input list; no loop rewrite."""
import ast
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import random
import textwrap

from common import ROOT, file_digest, write_json

TRAIN_FUNCTION_SHA256 = '388155630845e9c2eaacdf070f1a49ddda5c9a66f5643eb9332bf6480b25d8b6'
CANDIDATE_SHA256 = '51684631a97f2e4ff7cfaad5739918a84137c78c38e3885a9aadce31c21d9c38'


def validate_sampling(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {'mode', 'candidate_file', 'candidate_sha256', 'smoke_items'}:
        raise ValueError('Invalid sampling fields')
    if value.get('mode') not in ('control', 'resampled'):
        raise ValueError('sampling.mode must be control or resampled')
    if value.get('candidate_sha256') != CANDIDATE_SHA256:
        raise ValueError('Only the frozen reviewed v2 pool is supported')
    path = Path(value.get('candidate_file', ''))
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'reports':
        raise ValueError('Candidate path must be repository-relative under reports')
    smoke = value.get('smoke_items')
    if smoke is not None and (type(smoke) is not int or smoke < 2 or smoke % 2):
        raise ValueError('smoke_items must be an even positive sample count')
    return dict(value)


def id_hash(ids):
    return hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()


def make_plan(ids, labels, candidates, seed, mode):
    if mode not in ('control', 'resampled') or len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError('Invalid gradient IDs, labels or mode')
    if not candidates <= set(ids) or any(type(v) is not int or v not in (0, 1) for v in labels.values()):
        raise ValueError('Invalid candidate membership or labels')
    plans = [list(range(len(ids))), list(range(len(ids)))]
    removed = []
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    if mode == 'resampled':
        for label in (0, 1):
            hard = sorted(i for i in candidates if labels[i] == label)
            random.Random(seed).shuffle(hard)
            other = sorted(i for i in ids if labels[i] == label and i not in candidates)
            random.Random(seed + 1000003).shuffle(other)
            if len(hard) > len(other):
                raise ValueError('Insufficient distinct same-label noncandidates')
            displaced = other[:len(hard)]
            removed.extend(displaced)
            split = (len(hard) + 1) // 2
            for epoch, pairs in enumerate((zip(displaced[:split], hard[:split]), zip(displaced[split:], hard[split:]))):
                for old, new in pairs:
                    plans[epoch][index[old]] = index[new]
    counts = Counter(ids[i] for plan in plans for i in plan)
    if set(counts) != set(ids):
        raise ValueError('Lost joint coverage')
    for sample_id in ids:
        expected = 3 if mode == 'resampled' and sample_id in candidates else 1 if sample_id in removed else 2
        if counts[sample_id] != expected:
            raise ValueError('Exposure mismatch')
    classes = Counter(labels.values())
    if any(Counter(labels[ids[i]] for i in plan) != classes for plan in plans):
        raise ValueError('Per-epoch class imbalance')
    return plans, removed, counts


@contextmanager
def resample_training(upstream, options, gradient_ids, labels, report_dir):
    options = validate_sampling(options)
    if options is None:
        yield
        return
    path = ROOT / options['candidate_file']
    if file_digest(path) != options['candidate_sha256']:
        raise ValueError('Frozen candidate file changed')
    rows = json.loads(path.read_text())['rows']
    candidates = {r['sample_id'] for r in rows}
    if len(candidates) != 48 or len(rows) != 48 or not candidates <= set(gradient_ids):
        raise ValueError('Frozen candidates must be a subset of original gradient IDs')
    if any(labels[r['sample_id']] != r['label'] for r in rows):
        raise ValueError('Candidate original label mismatch')
    original = upstream.train_model
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != TRAIN_FUNCTION_SHA256:
        raise ValueError('Unsupported upstream train_model implementation')
    tree = ast.parse(source)
    hits = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'epoch_items']
    if len(hits) != 1 or ast.unparse(hits[0].value) != 'list(items)':
        raise ValueError('Unsupported upstream epoch input expression')
    hits[0].value = ast.parse('_reviewed_epoch_items(items, epoch)').body[0].value
    ast.fix_missing_locations(tree)
    signature = inspect.signature(original)

    @wraps(original)
    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        items = bound.arguments['items']
        config = bound.arguments['config']
        if len(items) != len(gradient_ids) or config.epochs != 2 or config.seed != 43 or bound.arguments.get('parallel', False):
            raise ValueError('Require two-epoch seed43 single-process original gradient ordering')
        active_ids = list(gradient_ids)
        active_items = items
        if options.get('smoke_items') is not None:
            selected = set(candidates)
            for label in (0, 1):
                needed = options['smoke_items'] // 2 - sum(labels[i] == label for i in selected)
                if needed < 0:
                    raise ValueError('Smoke subset too small for reviewed candidates')
                pool = [i for i in gradient_ids if labels[i] == label and i not in selected]
                if len(pool) < needed:
                    raise ValueError('Smoke subset exceeds gradient population')
                selected.update(pool[:needed])
            indices = [i for i, sample_id in enumerate(gradient_ids) if sample_id in selected]
            active_ids = [gradient_ids[i] for i in indices]
            active_items = [items[i] for i in indices]
        active_labels = {i: labels[i] for i in active_ids}
        plans, removed, exposures = make_plan(active_ids, active_labels, candidates, config.seed, options['mode'])
        ordered = []
        for epoch, plan in enumerate(plans):
            order = list(plan)
            random.Random(config.seed + epoch).shuffle(order)
            ordered.append([active_ids[i] for i in order])
        audit = {'state': 'running', 'options': options, 'gradient_count_before_smoke_subset': len(gradient_ids),
                 'actual_train_count': len(active_ids), 'actual_gradient_ids': active_ids,
                 'candidate_count': len(candidates), 'class_counts': dict(Counter(active_labels.values())),
                 'expected_updates': 2 * ((len(active_ids) + config.micro_batch - 1) // config.micro_batch),
                 'epoch_ordered_ids': ordered, 'epoch_ordered_id_sha256': [id_hash(order) for order in ordered],
                 'exposure_counts': dict(exposures), 'displaced_ids': removed, 'epochs_observed': [],
                 'upstream_function_sha256': TRAIN_FUNCTION_SHA256, 'test_accessed': False}
        audit_path = Path(report_dir) / 'sampling_audit.json'
        write_json(audit_path, audit)

        def epoch_items(current, epoch):
            if current is not active_items or epoch != len(audit['epochs_observed']) or epoch not in (0, 1):
                raise ValueError('Unexpected upstream epoch invocation')
            audit['epochs_observed'].append(epoch + 1)
            write_json(audit_path, audit)
            return [current[i] for i in plans[epoch]]

        # Copy *current* globals after CUDA guards are installed. Library hooks
        # live on referenced modules; only this function's one expression changes.
        namespace = dict(original.__globals__, _reviewed_epoch_items=epoch_items)
        exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
        cloned = namespace[original.__name__]
        bound.arguments['items'] = active_items
        try:
            result = cloned(*bound.args, **bound.kwargs)
            if audit['epochs_observed'] != [1, 2]:
                raise ValueError('Both planned epochs were not observed')
            audit['state'] = 'completed'
            return result
        except BaseException as exc:
            audit.update(state='failed', error=repr(exc))
            raise
        finally:
            write_json(audit_path, audit)

    upstream.train_model = train
    try:
        yield
    finally:
        upstream.train_model = original
