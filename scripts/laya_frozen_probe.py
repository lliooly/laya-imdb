"""Frozen pre-head representations and convex linear readout, with no data access."""
import math

import numpy as np


LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.)
LOCATIONS = ('encoder', 'post_head')


class MarkerCapture:
    """Capture raw encoder output and scorer input at the same actual markers."""
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.positions = None
        self.values = {}
        self.handles = []

    def __enter__(self):
        self.handles = [self.model.encoder.register_forward_hook(self._encoder),
                        self.model.scorer.register_forward_pre_hook(self._post_head)]
        return self

    def begin(self, positions):
        if self.values:
            raise RuntimeError('Previous batch captures were not consumed')
        if positions.ndim != 2 or positions.shape[1] != 2 or (positions < 0).any().item():
            raise ValueError('Expected two valid candidate marker positions')
        self.positions = positions

    def _save(self, name, candidates):
        if (self.model.training or self.torch.is_grad_enabled()
                or candidates.requires_grad or name in self.values):
            raise RuntimeError('Capture requires one frozen eval/inference forward per batch')
        if candidates.ndim != 3 or candidates.shape[1] != 2:
            raise ValueError('Expected batch x two candidates x hidden dimension')
        values = (candidates[:, 1]-candidates[:, 0]).float().cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite frozen features')
        self.values[name] = values

    def _encoder(self, module, args, output):
        if self.positions is None:
            raise RuntimeError('Set marker positions before forward')
        h = output.last_hidden_state
        if h.ndim != 3 or h.shape[0] != len(self.positions) or (self.positions >= h.shape[1]).any().item():
            raise ValueError('Encoder output and markers do not align')
        idx = self.positions[:, :, None].expand(-1, -1, h.shape[-1])
        self._save('encoder', self.torch.gather(h, 1, idx))

    def _post_head(self, module, args):
        self._save('post_head', args[0])

    def consume(self):
        if set(self.values) != set(LOCATIONS):
            raise RuntimeError('Both distinct extraction points must fire exactly once')
        result, self.values = self.values, {}
        self.positions = None
        return result

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def standardize(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite feature matrix')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    return mean, scale


def fit_readout(torch, features, labels, penalty, mean, scale):
    if penalty not in LAMBDAS:
        raise ValueError('Only predeclared regularization is allowed')
    x = torch.as_tensor((np.asarray(features, dtype=np.float64)-mean)/scale, dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    if y.shape != (len(x),) or not torch.isin(y, torch.tensor([0., 1.], dtype=torch.float64)).all().item():
        raise ValueError('Expected aligned binary training labels')
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, intercept], max_iter=400, max_eval=600,
        tolerance_grad=1e-8, tolerance_change=1e-12, history_size=50, line_search_fn='strong_wolfe')
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        logits = x @ weight + intercept
        loss = (torch.nn.functional.softplus(logits)-y*logits).mean() + penalty/2 * weight.square().sum()
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite probe objective')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in (weight, intercept)):
            raise FloatingPointError('Nonfinite probe gradients')
        evaluations += 1
        return loss

    optimizer.step(closure)
    objective = float(closure().detach())
    gradient = max(float(p.grad.abs().max()) for p in (weight, intercept))
    if gradient > 1e-5:
        raise RuntimeError(f'Probe not converged: max absolute gradient {gradient}')
    state = optimizer.state[weight]
    return {'lambda': penalty, 'weight': weight.detach().numpy().tolist(),
            'intercept': float(intercept.detach()), 'objective': objective,
            'max_abs_gradient': gradient, 'converged': True,
            'iterations': state['n_iter'], 'closure_evaluations': evaluations}


def logits(features, fit, mean, scale):
    z = ((np.asarray(features, dtype=np.float64)-mean)/scale) @ np.asarray(fit['weight']) + fit['intercept']
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite probe predictions')
    return z


def probability(z):
    p = .5*(1+np.tanh(np.asarray(z)/2))
    return np.column_stack((1-p, p))


def score(z, labels):
    y = np.asarray(labels)
    return {'count': len(y), 'correct': int(((np.asarray(z) > 0) == y).sum()),
            'accuracy': float(np.mean((np.asarray(z) > 0) == y)),
            'nll': float(np.mean(np.logaddexp(0., z)-y*z))}


def select_candidate(candidates):
    if not candidates:
        raise ValueError('No calibrated candidates')
    highest = max(c['calibration']['correct'] for c in candidates)
    tied = [c for c in candidates if c['calibration']['correct'] == highest]
    best_nll = min(c['calibration']['nll'] for c in tied)
    return max((c for c in tied if c['calibration']['nll'] <= best_nll+1e-12), key=lambda c: c['lambda'])


def paired_accuracy(labels, reference, candidate, repetitions=10000, seed=20261015):
    y = np.asarray(labels)
    a, b = np.asarray(reference) == y, np.asarray(candidate) == y
    difference = b.astype(float)-a.astype(float)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    draws = np.empty(repetitions)
    for i in range(repetitions):
        count = sum(difference[g[rng.integers(len(g), size=len(g))]].sum() for g in groups if len(g))
        draws[i] = count/len(y)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n = a_only+b_only
    # Log-gamma avoids overflow if the probe and full model disagree often.
    tail = sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2))
               for k in range(min(a_only, b_only)+1)) if n else .5
    return {'delta_accuracy_pp': float(100*difference.mean()),
            'paired_stratified_bootstrap95_pp': (100*np.quantile(draws, [.025, .975])).tolist(),
            'reference_only_correct': a_only, 'candidate_only_correct': b_only,
            'mcnemar_exact_nominal_two_sided_p': min(1., 2*tail),
            'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
            'uncertainty_scope': 'fixed fitted readouts and weights; sample uncertainty only'}
