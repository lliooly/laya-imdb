"""Shared binary metrics, stable temperature fitting and exact paired testing."""
import math
import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z, dtype=np.float64)))


def fit_temperature(margin, labels, bounds=(0.05, 20.0)):
    """NLL is convex in inverse temperature; bisect its derivative on fixed bounds."""
    margin, labels = np.asarray(margin, dtype=float), np.asarray(labels)
    lo, hi = 1 / bounds[1], 1 / bounds[0]
    derivative = lambda a: float(np.mean((sigmoid(a * margin) - labels) * margin))
    if derivative(lo) >= 0:
        return 1 / lo
    if derivative(hi) <= 0:
        return 1 / hi
    for _ in range(90):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 1 / ((lo + hi) / 2)


def auc_rank(labels, score):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def metrics(labels, margin, temperature=1.):
    labels, margin = np.asarray(labels, dtype=int), np.asarray(margin, dtype=float)
    assert len(labels) and len(labels) == len(margin) and np.isfinite(margin).all()
    pred = (margin > 0).astype(int)  # argmax picks negative on an exact tie
    z = margin / temperature
    p = sigmoid(z)
    nll = np.mean(np.logaddexp(0., np.where(labels == 1, -z, z)))
    cm = [[int(np.sum((labels == i) & (pred == j))) for j in range(2)] for i in range(2)]
    precision, recall, f1 = [], [], []
    for i in range(2):
        tp, fp, fn = cm[i][i], cm[1-i][i], cm[i][1-i]
        precision.append(tp / (tp + fp) if tp + fp else 0.)
        recall.append(tp / (tp + fn) if tp + fn else 0.)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    confidence = np.maximum(p, 1-p)
    correct = pred == labels
    bins = []
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        keep = (confidence >= low if low == 0 else confidence > low) & (confidence <= high)
        n = int(keep.sum())
        acc, conf = (float(correct[keep].mean()), float(confidence[keep].mean())) if n else (None, None)
        if n:
            ece += n / len(labels) * abs(acc-conf)
        bins.append({"lower": float(low), "upper": float(high), "count": n, "accuracy": acc, "confidence": conf})
    wrong = ~correct
    return {"count": len(labels), "accuracy": float(correct.mean()), "correct": int(correct.sum()),
            "macro_f1": float(np.mean(f1)), "precision_macro": float(np.mean(precision)),
            "recall_macro": float(np.mean(recall)), "precision_positive": precision[1], "recall_positive": recall[1],
            "auroc": auc_rank(labels, margin), "nll": float(nll),
            "brier_multiclass": float(2 * np.mean((p-labels)**2)), "brier_positive": float(np.mean((p-labels)**2)),
            "ece_15_equal_width_top_label": float(ece), "temperature": float(temperature),
            "confusion_matrix_actual_rows_predicted_columns": cm, "reliability_bins": bins,
            "error_mean_confidence": float(confidence[wrong].mean()) if wrong.any() else None,
            "errors_with_confidence_ge_095": int(np.sum(wrong & (confidence >= .95)))}


def paired(labels, margin_a, margin_b):
    ca = (np.asarray(margin_a) > 0) == labels
    cb = (np.asarray(margin_b) > 0) == labels
    a_only, b_only = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = a_only + b_only
    # Exact two-sided binomial McNemar test; this sum is stable for our small discordant counts.
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only)+1)) / 2**n) if n else 1.
    return {"a_only_correct": a_only, "b_only_correct": b_only, "both_correct": int(np.sum(ca & cb)),
            "both_wrong": int(np.sum(~ca & ~cb)), "mcnemar_exact_two_sided_p": p}
