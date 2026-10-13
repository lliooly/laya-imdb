"""Fixed decision-view algorithms; no fitting or selection on development labels."""
import copy
import hashlib
import json

import numpy as np

from comparison_metrics import metrics, paired, sigmoid

BUDGET = 466
WINDOWS = (128, 256, 466)
COEFFICIENT_BOUNDS = (-2., 2.)


def full_tokens(tok, texts):
    tok.backend_tokenizer.no_padding()
    tok.backend_tokenizer.no_truncation()
    return [e.ids for e in tok.backend_tokenizer.encode_batch(
        [s.replace(tok.mask_token, ' ') for s in texts], add_special_tokens=False)]


def window_tokens(tokens, rule, budget=BUDGET):
    if rule not in ('head', 'head_tail') or type(budget) is not int or budget < 2:
        raise ValueError('Invalid window rule or budget')
    if rule == 'head' or len(tokens) <= budget:
        return list(tokens[:budget])
    left = budget // 2
    return list(tokens[:left]) + list(tokens[-(budget-left):])


def canonical_logits(logits, order):
    """Slot s describes canonical option order[s], even on an exact tie."""
    values = np.asarray(logits, dtype=np.float64)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or list(sorted(order)) != [0, 1]):
        raise ValueError('Expected two finite logits and binary option permutation')
    result = np.empty(2, dtype=np.float64)
    result[order] = values
    return result


def probabilities(logits, temperature=1.):
    z = np.asarray(logits, dtype=np.float64)
    if (z.ndim != 2 or z.shape[1] != 2 or not len(z) or not np.isfinite(z).all()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Expected finite binary logits and positive temperature')
    positive = sigmoid((z[:, 1]-z[:, 0])/temperature)
    return np.column_stack((1-positive, positive))


def probability_margin(prob):
    p = np.asarray(prob, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1)):
        raise ValueError('Expected normalized binary probabilities')
    clipped = np.clip(p, 1e-15, 1-1e-15)
    return np.log(clipped[:, 1])-np.log(clipped[:, 0])


def gated_equal_fusion(head, tail, truncated):
    """Keep the exact prefix probabilities for all untruncated samples."""
    head, tail = np.asarray(head), np.asarray(tail)
    mask = np.asarray(truncated)
    probability_margin(head)
    probability_margin(tail)
    if head.shape != tail.shape or mask.dtype != bool or mask.shape != (len(head),):
        raise ValueError('Fusion views and truncation mask must align')
    if not np.array_equal(head[~mask], tail[~mask]):
        raise ValueError('Untruncated views must be bitwise identical')
    result = head.copy()
    result[mask] = (head[mask]+tail[mask])/2
    return result


def candidate_features(embeddings):
    """Equivariant mean/max cosine scores, excluding padded markers.

    Binary experiments supply exactly two candidates. Their mutual cosine is
    identical for each slot, so it cannot by itself change the ordering.
    """
    h = np.asarray(embeddings, dtype=np.float64)
    if h.ndim != 3 or h.shape[1] != 2 or not np.isfinite(h).all():
        raise ValueError('Expected finite [batch, 2, hidden] candidate representations')
    unit = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    scores = []
    for pool in (h.mean(axis=1), h.max(axis=1)):
        pool = pool / np.maximum(np.linalg.norm(pool, axis=-1, keepdims=True), 1e-12)
        scores.append(np.einsum('bkd,bd->bk', unit, pool))
    return (scores[0]+scores[1])/2


def fit_candidate_coefficient(logits, features, labels, temperature):
    """One bounded coefficient fitted to calibration NLL, never development."""
    z, f, y = np.asarray(logits), np.asarray(features), np.asarray(labels)
    probabilities(z, temperature)
    if f.shape != z.shape or not np.isfinite(f).all() or y.shape != (len(z),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Candidate calibration arrays differ')
    margin = (z[:, 1]-z[:, 0])/temperature
    feature = (f[:, 1]-f[:, 0])/temperature
    derivative = lambda a: float(np.mean((sigmoid(margin+a*feature)-y)*feature))
    lo, hi = COEFFICIENT_BOUNDS
    if np.all(feature == 0):
        coefficient = 0.
    elif derivative(lo) >= 0:
        coefficient = lo
    elif derivative(hi) <= 0:
        coefficient = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        coefficient = (lo+hi)/2
    before, after = metrics(y, margin), metrics(y, margin+coefficient*feature)
    return {'coefficient': float(coefficient), 'bounds': list(COEFFICIENT_BOUNDS),
            'temperature': float(temperature), 'fit_split': 'calibration_only',
            'calibration_count': len(y), 'before': before, 'after': after,
            'head_training_eligible': after['accuracy'] > before['accuracy'] and after['nll'] < before['nll'],
            'eligibility_scope': 'calibration signal only; not held-out evidence'}


def stratified_metrics(labels, prob, truncated):
    y, mask = np.asarray(labels), np.asarray(truncated, dtype=bool)
    margin = probability_margin(prob)
    if y.shape != margin.shape or mask.shape != margin.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Metric labels and truncation mask must align')
    report = {'all': metrics(y, margin)}
    for name, keep in (('truncated', mask), ('untruncated', ~mask)):
        report[name] = metrics(y[keep], margin[keep]) if keep.any() else {'count': 0}
    return report


def compare_views(ids, labels, before, after, truncated):
    y = np.asarray(labels)
    ma, mb = probability_margin(before), probability_margin(after)
    mask = np.asarray(truncated, dtype=bool)
    if len(ids) != len(y) or len(ids) != len(set(ids)) or ma.shape != y.shape or mb.shape != y.shape or mask.shape != y.shape:
        raise ValueError('Paired views must have unique aligned IDs')
    a, b = ma > 0, mb > 0
    flips = a != b
    return {'agreement_rate': float((~flips).mean()), 'flip_count': int(flips.sum()),
            'flip_rate': float(flips.mean()), 'flipped_ids': [i for i, flag in zip(ids, flips) if flag],
            'flips_truncated': int((flips & mask).sum()),
            'flips_untruncated': int((flips & ~mask).sum()),
            'corrected_ids': [i for i, flag in zip(ids, (a != y) & (b == y)) if flag],
            'regressed_ids': [i for i, flag in zip(ids, (a == y) & (b != y)) if flag],
            'paired': paired(y, ma, mb),
            'paired_truncated': paired(y[mask], ma[mask], mb[mask]) if mask.any() else None}


def sampled_rule(seed, epoch, sample_id):
    """Independent RNG stream, deterministic per original ID and epoch."""
    if type(seed) is not int or type(epoch) is not int or epoch < 0 or not isinstance(sample_id, str):
        raise ValueError('Invalid window schedule identity')
    payload = json.dumps([seed, epoch, sample_id], separators=(',', ':')).encode()
    return 'head_tail' if hashlib.sha256(payload).digest()[0] < 128 else 'head'


def validate_window_training(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'mode'} or value['mode'] not in ('prefix', 'random_head_tail'):
        raise ValueError('window_training requires only mode=prefix or random_head_tail')
    return copy.deepcopy(value)
