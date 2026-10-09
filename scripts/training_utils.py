"""Deterministic scheduling, metrics and checkpoint ranking."""
import math
import numpy as np


def learning_rate(step, total_steps, peak, warmup_fraction):
    """step is the zero-based update index; last update has zero LR."""
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    number = step + 1
    if number <= warmup:
        return peak * number / warmup
    return peak * max(0, total_steps - number) / max(1, total_steps - warmup)


def classification_metrics(labels, predictions, loss_sum):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if len(labels) == 0 or labels.shape != predictions.shape:
        raise ValueError("Nonempty aligned labels and predictions required")
    cm = [[int(((labels == i) & (predictions == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    f1 = [2 * cm[i][i] / max(1, 2 * cm[i][i] + cm[1-i][i] + cm[i][1-i]) for i in (0, 1)]
    return {"count": len(labels), "loss": loss_sum / len(labels),
            "accuracy": float((labels == predictions).mean()), "macro_f1": sum(f1)/2,
            "confusion_matrix_rows_true": cm}


def is_better(candidate, best):
    return best is None or (candidate["macro_f1"], -candidate["loss"]) > (best["macro_f1"], -best["loss"])


def epoch_order(count, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(count)
