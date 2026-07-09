"""
shap_analysis/figures.py
------------------------
Attribution figures, all fed from the significance summaries / per-fold frames
built in tables.py. Every bar carries a cross-fold error bar (± std over the 5
folds) so the figures inherit the same significance discipline as the tables.

The modality colour palette is defined once here and shared by every figure
(and re-exported for the spatial / noise scripts).

Figures
-------
fig_phi_ps_marginal   : Fig 3 — grouped bars phi / PS_raw / phi_grand (± std).
fig_scatter_phi_vs_ps : Fig 4 — scatter phi vs PS_raw, one point per (modality,fold).
fig_phi_decomposition : Fig 5 — stacked phi = phi_grand + phi_small (± std).
fig_geo_redistribution: CoM vs CoM+geo phi redistribution (± std).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single shared modality palette (matches paper/README.md).
MODALITY_COLORS = {
    "S2_S2VI": "#4e79a7", "S1": "#f28e2b", "weather": "#e15759",
    "DEM": "#76b7b2", "geo": "#59a14f",
}


def _save(fig, out) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _mean_std(summary, views, base):
    """(means, stds) for value ``base`` in the natural view order."""
    idx = summary.set_index("modality")
    means = np.array([idx.loc[v, f"{base}_mean"] if v in idx.index else np.nan for v in views])
    stds  = np.array([idx.loc[v, f"{base}_std"]  if v in idx.index else np.nan for v in views])
    return means, stds


def fig_phi_ps_marginal(summary, views, out, metric="f1_weighted"):
    """Fig 3 — grouped bars: phi vs PS_raw vs phi_grand, with ± std error bars."""
    colors = [MODALITY_COLORS[v] for v in views]
    phi_m, phi_s   = _mean_std(summary, views, "phi")
    ps_m,  ps_s    = _mean_std(summary, views, "PS_raw")
    pg_m,  pg_s    = _mean_std(summary, views, "phi_grand")

    x, w = np.arange(len(views)), 0.25
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w, phi_m, w, yerr=phi_s, capsize=3, color=colors, alpha=1.0, label="phi (Shapley)")
    ax.bar(x,     ps_m,  w, yerr=ps_s,  capsize=3, color=colors, alpha=0.6, label="PS_raw")
    ax.bar(x + w, pg_m,  w, yerr=pg_s,  capsize=3, color=colors, alpha=0.3, label="phi_grand")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(views)
    ax.set_ylabel(f"{metric} contribution")
    ax.set_title("phi vs PS_raw vs phi_grand (mean ± std, 5 folds)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_scatter_phi_vs_ps(per_fold, views, out):
    """Fig 4 — scatter phi(m) vs PS_raw(m), one point per (modality × fold)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for v in views:
        sub = per_fold[per_fold["modality"] == v]
        ax.scatter(sub["phi"], sub["PS_raw"], color=MODALITY_COLORS[v], label=v, s=40)
    both = per_fold[["phi", "PS_raw"]].to_numpy()
    both = both[~np.isnan(both).any(axis=1)]
    if both.size:
        lo, hi = both.min() - 0.01, both.max() + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel("phi (Shapley)"); ax.set_ylabel("PS_raw")
    ax.set_title("phi vs PS_raw (per modality × fold)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_phi_decomposition(summary, views, out):
    """Fig 5 — stacked bars phi = phi_grand + phi_small, with ± std on each part."""
    colors = [MODALITY_COLORS[v] for v in views]
    pg_m, pg_s = _mean_std(summary, views, "phi_grand")
    psm_m, psm_s = _mean_std(summary, views, "phi_small")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(views, pg_m, yerr=pg_s, capsize=3, color=colors, alpha=1.0, label="phi_grand")
    ax.bar(views, psm_m, bottom=pg_m, yerr=psm_s, capsize=3, color=colors, alpha=0.4,
           hatch="//", label="phi_small")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("phi decomposition")
    ax.set_title("phi = phi_grand + phi_small (mean ± std, 5 folds)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_geo_redistribution(geo_summary, views, out):
    """CoM vs CoM+geo phi per modality (mean ± std), for the geo-effect comparison."""
    colors = [MODALITY_COLORS[v] for v in views]
    com_m, com_s = _mean_std(geo_summary, views, "phi_com")
    geo_m, geo_s = _mean_std(geo_summary, views, "phi_geo")

    x, w = np.arange(len(views)), 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w/2, com_m, w, yerr=com_s, capsize=3, color=colors, alpha=1.0, label="CoM")
    ax.bar(x + w/2, geo_m, w, yerr=geo_s, capsize=3, color=colors, alpha=0.5, label="CoM+geo")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(views)
    ax.set_ylabel("phi (Shapley)")
    ax.set_title("phi redistribution: CoM vs CoM+geo (mean ± std)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)
