"""
shap_analysis/stats.py
----------------------
Cross-fold significance layer — the transversal statistics wired once so that
every table and figure reports mean +/- std with a signal-to-noise gate, and
model comparisons carry a paired test.

Design
------
The v-dicts (source of truth) give, per fold, one value for each quantity
(phi, PS, v(S), ...). This module turns a *set of per-fold values* into a
resolved/unresolved verdict:

    SNR = mean / std   over the K folds
    resolved  <=>  |SNR| > SNR_THRESHOLD

The same |SNR| > 2 gate already used for the Interaction Index in the appendix
is promoted here to a single constant so it applies everywhere identically.
Anything below the gate is reported as "not resolved" (n.r.), never as a result.

Public API
----------
SNR_THRESHOLD    : the single |SNR| gate (2.0).
agg              : per-quantity {mean, std, snr, n, resolved}.
agg_frame        : tidy DataFrame -> mean/std/snr/resolved columns per value.
fmt              : "mean +/- std" string, or "n.r." when unresolved.
wilcoxon_paired  : paired Wilcoxon signed-rank test across folds (model A vs B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# The single significance gate, reused everywhere (tables, figures, appendix SII).
SNR_THRESHOLD = 2.0


def agg(values, ddof: int = 1) -> dict:
    """
    Aggregate a set of per-fold values into a significance verdict.

    Args:
        values: iterable of per-fold scalars (NaNs are ignored).
        ddof:   delta degrees of freedom for std (1 = sample std, matches pandas).

    Returns:
        {mean, std, snr, n, resolved}. snr = mean/std; resolved = |snr| > SNR_THRESHOLD.
        A zero/degenerate std yields snr = +/-inf (resolved) if mean != 0, else nan.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "snr": np.nan, "n": 0, "resolved": False}

    mean = float(arr.mean())
    std = float(arr.std(ddof=ddof)) if n > 1 else 0.0

    if std == 0.0:
        snr = np.inf * np.sign(mean) if mean != 0 else np.nan
    else:
        snr = mean / std

    resolved = bool(np.isfinite(snr) and abs(snr) > SNR_THRESHOLD) or (std == 0.0 and mean != 0)
    return {"mean": mean, "std": std, "snr": snr, "n": int(n), "resolved": resolved}


def agg_frame(df: pd.DataFrame, group_cols, value_cols) -> pd.DataFrame:
    """
    Collapse per-fold rows into per-group significance rows.

    For each value column ``v``, emits ``v_mean``, ``v_std``, ``v_snr`` and
    ``v_resolved`` (bool). Grouping is typically by (model, modality) or
    (model, coalition), with one input row per fold.

    Args:
        df:          long-form DataFrame with one row per (group..., fold).
        group_cols:  columns identifying a group (e.g. ["model", "modality"]).
        value_cols:  columns to aggregate across folds.

    Returns:
        One row per group with the *_mean/_std/_snr/_resolved columns, plus
        ``n_folds``.
    """
    group_cols = list(group_cols)
    value_cols = list(value_cols)
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        n_folds = 0
        for col in value_cols:
            a = agg(grp[col].values)
            row[f"{col}_mean"] = a["mean"]
            row[f"{col}_std"] = a["std"]
            row[f"{col}_snr"] = a["snr"]
            row[f"{col}_resolved"] = a["resolved"]
            n_folds = max(n_folds, a["n"])
        row["n_folds"] = n_folds
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, std, snr=None, decimals: int = 3, unresolved: str = "n.r.",
        gate: bool = True, sep: str = " +/- ") -> str:
    """
    Render a "mean +/- std" cell, or ``unresolved`` when |SNR| <= threshold.

    Args:
        mean, std: the aggregated statistics.
        snr:       signal-to-noise; if None it is recomputed as mean/std.
        decimals:  rounding for both mean and std.
        unresolved:string returned when the value fails the SNR gate.
        gate:      if False, always render the number (no n.r. substitution).
        sep:       separator between mean and std (use r" $\\pm$ " for LaTeX).

    Returns:
        e.g. "0.601 +/- 0.012", or "n.r." when unresolved.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if snr is None:
        snr = (mean / std) if (std not in (0, None) and not np.isnan(std)) else np.inf
    if gate and not (np.isfinite(snr) and abs(snr) > SNR_THRESHOLD) and not (std == 0 and mean != 0):
        return unresolved
    return f"{mean:.{decimals}f}{sep}{std:.{decimals}f}"


def wilcoxon_paired(a_folds, b_folds, alternative: str = "two-sided") -> dict:
    """
    Paired Wilcoxon signed-rank test between two models across folds.

    Used for model comparisons — in particular the geo effect on phi(S1)
    (CoM vs CoM+geo). Pairs are matched by fold order.

    NOTE (low power): with K=5 folds the minimum achievable two-sided p-value
    is 0.0625, so a "significant" verdict is impossible at alpha=0.05. Report the
    p-value and the paired differences; treat this as directional evidence, and
    only present the effect as a result if the test *holds* (small p, consistent
    sign). This mirrors the manifesto's guidance on phi(S1).

    Args:
        a_folds, b_folds: per-fold values for model A and model B (same order).
        alternative:      passed to scipy.stats.wilcoxon.

    Returns:
        {stat, p, n, mean_diff, median_diff, all_same_sign, note}. When every
        paired difference is zero or n < 1, stat/p are NaN.
    """
    a = np.asarray(list(a_folds), dtype=float)
    b = np.asarray(list(b_folds), dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    n = diff.size

    note = ""
    if n < 1 or np.allclose(diff, 0.0):
        stat, p = np.nan, np.nan
        note = "no non-zero differences"
    else:
        try:
            res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        except ValueError as exc:  # e.g. all differences zero after zero_method
            stat, p = np.nan, np.nan
            note = str(exc)

    nonzero = diff[diff != 0]
    all_same_sign = bool(nonzero.size > 0 and np.all(np.sign(nonzero) == np.sign(nonzero[0])))
    return {
        "stat": stat, "p": p, "n": int(n),
        "mean_diff": float(np.mean(diff)) if n else np.nan,
        "median_diff": float(np.median(diff)) if n else np.nan,
        "all_same_sign": all_same_sign,
        "note": note,
    }
