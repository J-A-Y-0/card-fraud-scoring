"""Evaluation measures for a fixed review budget."""

import numpy as np


def fdr_at(y_true, score, rate: float = 0.03) -> float:
    """Fraud detection rate: share of all frauds found in the top `rate` of records by score."""
    y = np.asarray(y_true)
    s = np.asarray(score)
    if y.sum() == 0:
        return float("nan")
    k = int(round(len(y) * rate))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against decile bins of `expected`.

    Rule of thumb: < 0.1 stable, 0.1-0.25 drifting, > 0.25 shifted.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    pe = np.bincount(np.searchsorted(edges, e, side="right"), minlength=len(edges) + 1) / len(e)
    pa = np.bincount(np.searchsorted(edges, a, side="right"), minlength=len(edges) + 1) / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def ks_statistic(x, y) -> float:
    """Univariate KS between the fraud and non-fraud distributions of x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    cum_bad = np.cumsum(ys) / max(ys.sum(), 1)
    cum_good = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    # Evaluate only at the last occurrence of each distinct value so ties don't count.
    last = np.r_[xs[1:] != xs[:-1], True]
    return float(np.max(np.abs(cum_bad[last] - cum_good[last])))
