"""Train-only sparse sentiment features and calibration-only convex fusion."""
import re

import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import clean_text
from comparison_metrics import sigmoid


TOKEN_PATTERN = r"(?u)\b\w+(?:['’]\w+)*\b(?:/\d+(?:\.\d+)?)?"
KINDS = ('tfidf_lr', 'nb_ngram_lr')


def checked_labels(labels, count):
    y = np.asarray(labels)
    if y.shape != (count,) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Require aligned labels and both binary classes')
    return y.astype(int)


def nb_log_count_ratio(binary_features, labels):
    y = checked_labels(labels, binary_features.shape[0])
    if binary_features.nnz and not np.all(binary_features.data == 1):
        raise ValueError('NB document-frequency weighting requires binary features')
    pos = np.asarray(binary_features[y == 1].sum(axis=0)).ravel() + 1.
    neg = np.asarray(binary_features[y == 0].sum(axis=0)).ravel() + 1.
    return np.log(pos / pos.sum()) - np.log(neg / neg.sum())


def make_vectorizer(kind, min_df=3, max_features=300000):
    common = dict(lowercase=True, strip_accents=None, stop_words=None,
                  token_pattern=TOKEN_PATTERN, min_df=min_df,
                  max_features=max_features, dtype=np.float64)
    if kind == 'tfidf_lr':
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm='l2', **common)
    if kind == 'nb_ngram_lr':
        return CountVectorizer(ngram_range=(1, 3), binary=True, **common)
    raise ValueError('Unknown sparse branch')


def transform(bundle, texts):
    x = bundle['vectorizer'].transform([clean_text(t) for t in texts])
    if bundle['nb_ratio'] is not None:
        x = x.multiply(bundle['nb_ratio']).tocsr()
    return x


def train_branch(kind, texts, labels):
    y = checked_labels(labels, len(texts))
    vectorizer = make_vectorizer(kind)
    x = vectorizer.fit_transform([clean_text(t) for t in texts])
    ratio = nb_log_count_ratio(x, y) if kind == 'nb_ngram_lr' else None
    if ratio is not None:
        x = x.multiply(ratio).tocsr()
    classifier = LogisticRegression(C=4. if kind == 'tfidf_lr' else 1.,
        penalty='l2', solver='liblinear', tol=1e-5, max_iter=1000,
        random_state=20261018)
    classifier.fit(x, y)
    if not np.array_equal(classifier.classes_, [0, 1]) or not np.isfinite(classifier.coef_).all():
        raise ValueError('Invalid classifier')
    return dict(kind=kind, vectorizer=vectorizer, nb_ratio=ratio, classifier=classifier), dict(
        feature_count=int(x.shape[1]), train_nonzero=int(x.nnz),
        iterations=classifier.n_iter_.tolist(), C=classifier.C,
        train_count=len(y), ngram_range=list(vectorizer.ngram_range),
        min_df=vectorizer.min_df, max_features=vectorizer.max_features)


def decision(bundle, texts):
    values = bundle['classifier'].decision_function(transform(bundle, texts))
    if values.shape != (len(texts),) or not np.isfinite(values).all():
        raise ValueError('Invalid sparse margins')
    return values


def fit_sigmoid(margins, labels, regularization=.001):
    z = np.asarray(margins, dtype=float)
    y = checked_labels(labels, len(z))
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError('Invalid calibration margins')

    def objective(ab):
        a, b = ab
        logits = a*z+b
        residual = sigmoid(logits)-y
        penalty = regularization*((a-1)**2+b*b)
        loss = np.mean(np.logaddexp(0., logits)-y*logits)+penalty
        gradient = [np.mean(residual*z)+2*regularization*(a-1),
                    np.mean(residual)+2*regularization*b]
        return float(loss), np.asarray(gradient)

    result = minimize(objective, [1., 0.], jac=True, method='L-BFGS-B',
        bounds=[(.05, 20.), (-5., 5.)], options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 1000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Sigmoid calibration did not converge: '+str(result.message))
    a, b = result.x
    return dict(slope=float(a), intercept=float(b), regularization=regularization,
        bounds=[[.05, 20.], [-5., 5.]], objective=float(result.fun),
        optimizer_message=str(result.message), iterations=int(result.nit), fit_count=len(y))


def calibrated_probability(margins, fit):
    return sigmoid(fit['slope']*np.asarray(margins)+fit['intercept'])


def probability_nll(labels, probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1-1e-12)
    y = checked_labels(labels, len(p))
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def mix(neural, sparse, weight):
    a, b = np.asarray(neural, dtype=float), np.asarray(sparse, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not 0 <= weight <= .25:
        raise ValueError('Mixture shape or weight invalid')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any() or (a > 1).any() or (b > 1).any():
        raise ValueError('Invalid probabilities')
    return (1-weight)*a+weight*b


def fit_sparse_weight(neural, sparse, labels):
    a, b = np.asarray(neural), np.asarray(sparse)
    mix(a, b, 0.)
    y = checked_labels(labels, len(a))
    d = b-a

    def derivative(weight):
        p = np.clip(mix(a, b, weight), 1e-12, 1-1e-12)
        return float(np.mean((p-y)*d/(p*(1-p))))

    lo, hi = 0., .25
    if np.array_equal(a, b) or derivative(lo) >= 0:
        weight = lo
    elif derivative(hi) <= 0:
        weight = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        weight = (lo+hi)/2
    return dict(sparse_weight=float(weight), neural_weight=float(1-weight),
        bounds=[0., .25], fit_split='fixed_calibration_400', method='convex_nll_bisection_80',
        calibration_nll=probability_nll(y, mix(a, b, weight)),
        derivative_at_zero=derivative(0.), derivative_at_upper=derivative(.25),
        boundary_hit=weight in (0., .25))


def validate_partitions(gradient_ids, calibration_ids, development_ids):
    parts = [list(v) for v in (gradient_ids, calibration_ids, development_ids)]
    if any(len(v) != len(set(v)) for v in parts):
        raise ValueError('Duplicate partition IDs')
    sets = [set(v) for v in parts]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Partitions overlap')
    return sets
