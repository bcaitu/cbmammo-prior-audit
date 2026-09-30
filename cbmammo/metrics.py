"""Metrics with patient-clustered bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def safe_auc(y, s):
    """Binary AUC from P(class 1); one-vs-rest macro AUC for multiclass score matrices."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    s = np.asarray(s)
    if s.ndim == 2 and s.shape[1] > 2:
        present = np.unique(y)
        if len(present) < s.shape[1]:
            s = s[:, present] / s[:, present].sum(1, keepdims=True).clip(1e-9)
            y = np.searchsorted(present, y)
        if s.shape[1] == 2:
            return roc_auc_score(y, s[:, 1])
        return roc_auc_score(y, s, multi_class="ovr", average="macro")
    if s.ndim == 2:
        s = s[:, 1]
    return roc_auc_score(y, s)


def cluster_bootstrap(fn, y, pred, groups, n=1000, seed=0):
    """Point estimate + 95% CI, resampling PATIENTS (not images) with replacement."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    point = fn(y, pred)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        gs = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in gs])
        try:
            v = fn(y[ii], pred[ii])
        except ValueError:
            continue
        if v == v:
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def paired_bootstrap_diff(fn, y, pa, pb, groups, n=1000, seed=0):
    """CI and two-sided p for metric(A) - metric(B) on the same samples (patient-clustered)."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    d0 = fn(y, pa) - fn(y, pb)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ii = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        try:
            ds.append(fn(y[ii], pa[ii]) - fn(y[ii], pb[ii]))
        except ValueError:
            pass
    ds = np.array([d for d in ds if d == d])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean()) if len(ds) else np.nan
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(min(p, 1.0))


def tost_equivalence(ci_lo, ci_hi, margin):
    """Equivalence at +-margin is shown when the 90%/95% CI of the difference lies inside it."""
    return bool(ci_lo > -margin and ci_hi < margin)
