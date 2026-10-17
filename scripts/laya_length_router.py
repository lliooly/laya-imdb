"""Fixed, label-free dispatch to the frozen 512/2048 context models."""
from numbers import Integral

import numpy as np

REVIEW_BUDGET_512 = 466


def routes(token_lengths):
    """Lengths are full review tokens from the frozen tokenizer, before truncation."""
    lengths = list(token_lengths)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in lengths):
        raise ValueError('Full review token lengths must be nonnegative integers')
    return np.asarray([2048 if n > REVIEW_BUDGET_512 else 512 for n in lengths], dtype=np.int64)


def route_predict(token_lengths, predict_512, predict_2048):
    """Callbacks receive original indices and return logits in that same order.

    Each callback should batch its assigned samples using the corresponding
    frozen model and its original encoding protocol. An empty branch is skipped.
    The selector itself never examines labels or logits to choose a model.
    """
    chosen = routes(token_lengths)
    output = np.empty((len(chosen), 2), dtype=np.float64)
    for cap, predict in ((512, predict_512), (2048, predict_2048)):
        indices = np.flatnonzero(chosen == cap)
        if not len(indices):
            continue
        value = np.asarray(predict(indices.tolist()), dtype=np.float64)
        if value.shape != (len(indices), 2) or not np.isfinite(value).all():
            raise ValueError('Each branch must return one finite binary-logit row per assigned sample')
        output[indices] = value
    return output, chosen
