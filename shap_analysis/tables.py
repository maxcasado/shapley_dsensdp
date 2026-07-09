"""
shap_analysis/tables.py
-----------------------
Deliverable tables, all derived from the single source (the v-dicts) with the
significance layer (mean +/- std + SNR gate) baked in. Every builder returns
both the per-fold long form and a per-group significance summary, so nothing is
ever reported as a bare cross-fold mean.

Builders
--------
attribution_table    : main attribution table (ex-Table 2) — phi, PS, phi_grand,
                       phi_small, v_single per modality, each mean +/- std + SNR.
coalition_value_table: the 16 (resp. 32) v(S) with mean +/- std and the solo lift
                       v({m}) - v(∅) — the transparent-to-baseline replacement for
                       the dead "40%" of the abstract (new, was missing).
geo_comparison_table : CoM vs CoM+geo in fixed-views (ex-Table 3), with Delta, std,
                       and the paired Wilcoxon test on phi (esp. phi(S1)).

to_latex             : render a significance summary as a LaTeX table (n.r.-gated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comparison import build_shapley_table, add_phi_decomposition
from .shapley import scalar_vdict
from .characteristic import DEFAULT_METRIC
from .stats import agg_frame, fmt, wilcoxon_paired


# ── Main attribution table (ex-Table 2) ───────────────────────────────────────

ATTRIBUTION_VALUE_COLS = ["phi", "PS_raw", "PS_model", "v_single",
                          "marginal_grand", "phi_grand", "phi_small"]


def attribution_table(v_dicts: dict, ps_df: pd.DataFrame, view_names: list,
                      metric: str = DEFAULT_METRIC, fixed_views: list = None) -> dict:
    """
    Per-modality attribution table with cross-fold significance.

    Args:
        v_dicts:     {fold_id -> v_dict} (v1 or v2).
        ps_df:       concatenated perceptual_score_fold*.csv (cols modality, fold_id,
                     PS_raw, PS_model). Pass an empty/None frame to skip PS columns.
        view_names:  full list of views (incl. fixed).
        metric:      characteristic function (post-processing flag).
        fixed_views: views always present (phi=0, no marginal_grand).

    Returns:
        {"per_fold": long DataFrame, "summary": significance DataFrame}. The
        summary carries <col>_mean/_std/_snr/_resolved for each ATTRIBUTION_VALUE_COLS
        plus v_empty (the baseline v(∅), same across modalities), and v_full.
    """
    fixed_views = fixed_views or []
    df = add_phi_decomposition(build_shapley_table(v_dicts, view_names, metric,
                                                    fixed_views=fixed_views))

    if ps_df is not None and len(ps_df):
        ps = ps_df[["fold_id", "modality", "PS_raw", "PS_model"]]
        df = df.merge(ps, on=["fold_id", "modality"], how="left")
    else:
        df["PS_raw"] = np.nan
        df["PS_model"] = np.nan

    summary = agg_frame(df, ["modality"], ATTRIBUTION_VALUE_COLS)

    # v_empty / v_full are per-fold-constant; carry their mean for reference.
    ref = agg_frame(df, ["modality"], ["v_empty", "v_full"])
    summary = summary.merge(ref[["modality", "v_empty_mean", "v_full_mean"]],
                            on="modality", how="left")
    # Preserve the natural view order.
    order = {m: i for i, m in enumerate(view_names)}
    summary = summary.sort_values("modality", key=lambda s: s.map(order)).reset_index(drop=True)
    return {"per_fold": df, "summary": summary}


# ── Coalition-value table (new) ───────────────────────────────────────────────

def coalition_value_table(v_dicts: dict, view_names: list,
                          metric: str = DEFAULT_METRIC) -> dict:
    """
    All 2^N coalition values v(S), mean +/- std across folds, with the solo lift.

    The solo lift v({m}) - v(∅) is reported transparently against the baseline,
    which is the honest replacement for the old "40%" framing.

    Returns:
        {"per_fold": long DataFrame, "summary": significance DataFrame}. Coalitions
        are labelled by sorted "+"-joined view names ("(empty)" for v(∅)).
    """
    rows = []
    for fold_id, v_dict in v_dicts.items():
        v = scalar_vdict(v_dict, metric)
        v_empty = v.get(frozenset(), np.nan)
        for S, val in v.items():
            rows.append({
                "fold_id": fold_id,
                "coalition": "+".join(sorted(S)) if len(S) else "(empty)",
                "size": len(S),
                "v": val,
                "lift": val - v_empty,
            })
    df = pd.DataFrame(rows)
    summary = agg_frame(df, ["coalition", "size"], ["v", "lift"])
    summary = summary.sort_values(["size", "coalition"]).reset_index(drop=True)
    return {"per_fold": df, "summary": summary}


# ── Geo comparison (ex-Table 3) with paired Wilcoxon ──────────────────────────

def geo_comparison_table(v_dicts_com: dict, v_dicts_geo: dict, common_views: list,
                         metric: str = DEFAULT_METRIC,
                         geo_view_names: list = None) -> dict:
    """
    CoM vs CoM+geo (fixed=geo) attribution comparison with a paired Wilcoxon test.

    For each shared modality: phi under CoM and under CoM+geo (mean +/- std), the
    per-fold Delta = phi_geo - phi_com (mean +/- std), and the paired Wilcoxon
    signed-rank test across folds. Present phi(S1) rising with geo as a *result*
    only if the test holds (see stats.wilcoxon_paired's low-power caveat).

    Returns:
        {"per_fold": long DataFrame (model, fold_id, modality, phi),
         "summary": one row per modality with phi_com/phi_geo/delta stats and
                    wilcoxon stat/p/all_same_sign}.
    """
    geo_view_names = geo_view_names or (list(common_views) + ["geo"])

    phi_com = build_shapley_table(v_dicts_com, common_views, metric, fixed_views=[])
    phi_geo = build_shapley_table(v_dicts_geo, geo_view_names, metric, fixed_views=["geo"])

    com = phi_com[phi_com["modality"].isin(common_views)][["fold_id", "modality", "phi"]].copy()
    geo = phi_geo[phi_geo["modality"].isin(common_views)][["fold_id", "modality", "phi"]].copy()
    com["model"] = "com"
    geo["model"] = "com_geo"
    per_fold = pd.concat([com, geo], ignore_index=True)

    merged = com.merge(geo, on=["fold_id", "modality"], suffixes=("_com", "_geo"))
    merged["delta"] = merged["phi_geo"] - merged["phi_com"]

    stats_df = agg_frame(merged, ["modality"], ["phi_com", "phi_geo", "delta"])

    wil = []
    for m in common_views:
        sub = merged[merged["modality"] == m].sort_values("fold_id")
        w = wilcoxon_paired(sub["phi_geo"].values, sub["phi_com"].values)
        wil.append({"modality": m, "wilcoxon_stat": w["stat"], "wilcoxon_p": w["p"],
                    "wilcoxon_n": w["n"], "all_same_sign": w["all_same_sign"]})
    summary = stats_df.merge(pd.DataFrame(wil), on="modality", how="left")
    order = {m: i for i, m in enumerate(common_views)}
    summary = summary.sort_values("modality", key=lambda s: s.map(order)).reset_index(drop=True)
    return {"per_fold": per_fold, "summary": summary}


# ── LaTeX rendering ───────────────────────────────────────────────────────────

def to_latex(summary: pd.DataFrame, value_cols: list, label_col: str,
             col_labels: dict = None, caption: str = "", label: str = "",
             gate: bool = True, decimals: int = 3, star_table: bool = False) -> str:
    """
    Render a significance summary to LaTeX, one "mean +/- std" cell per value
    (or "n.r." when the SNR gate fails and ``gate`` is True).

    Args:
        summary:    DataFrame with <col>_mean/_std/_snr for each value col.
        value_cols: base names (e.g. "phi"); expects "<col>_mean" etc. present.
        label_col:  the row-label column (e.g. "modality" or "coalition").
        col_labels: pretty header per value col (defaults to the raw name).
        gate:       apply the |SNR|>2 n.r. gate.
        star_table: use table* (two-column spanning) instead of table.
    """
    col_labels = col_labels or {}
    env = "table*" if star_table else "table"
    align = "l" + "c" * len(value_cols)
    esc = lambda s: str(s).replace("_", r"\_")
    header = f"{esc(label_col)} & " + " & ".join(col_labels.get(c, esc(c)) for c in value_cols) + r" \\"

    lines = [
        f"\\begin{{{env}}}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}", "\\toprule", header, "\\midrule",
    ]
    for _, r in summary.iterrows():
        cells = []
        for c in value_cols:
            snr = r.get(f"{c}_snr", None)
            cells.append(fmt(r.get(f"{c}_mean"), r.get(f"{c}_std"), snr,
                             decimals=decimals, gate=gate, sep=r" $\pm$ "))
        lines.append(f"{esc(r[label_col])} & " + " & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines)
