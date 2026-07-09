"""
shap_analysis/comparison.py
-----------------------------
Build the Shapley/Perceptual-Score comparison tables (Steps 1, 3, 4 of the
phi vs PS experiment): per-fold Shapley decomposition, phi_grand/phi_small
decomposition, pairwise redundancy, and correlation diagnostics.
"""
import math
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .shapley import shapley_values, scalar_vdict


def load_v_dicts(v_dict_dir: Path, fold_ids: list) -> dict:
    """{fold_id: v_dict} from v_dict_fold{N}.pkl files."""
    v_dicts = {}
    for fold_id in fold_ids:
        path = Path(v_dict_dir) / f"v_dict_fold{fold_id}.pkl"
        with open(path, "rb") as f:
            v_dicts[fold_id] = pickle.load(f)
    return v_dicts


def build_shapley_table(
    v_dicts: dict,
    view_names: list,
    metric: str,
    fixed_views: list = None,
) -> pd.DataFrame:
    """
    Per-fold, per-modality table of Shapley quantities.

    Columns: fold_id, modality, phi, v_single, v_complement, marginal_grand,
             v_full, v_empty, n_free.
    For fixed views: v_single = v_complement = v_empty (baseline), and
    marginal_grand/phi are not defined within the fixed-views framework
    (marginal_grand -> NaN; phi is reported as 0, matching the Shapley
    convention used in run_subset_inference/shapley_values).
    """
    fixed_views = fixed_views or []
    free_views  = [v for v in view_names if v not in fixed_views]
    fixed_set   = frozenset(fixed_views)
    n_free      = len(free_views)

    rows = []
    for fold_id, v_dict in v_dicts.items():
        v_scalar = scalar_vdict(v_dict, metric)
        phi = shapley_values(view_names, v_scalar, fixed_views=fixed_views)

        v_empty = v_scalar.get(fixed_set, v_scalar.get(frozenset(), np.nan))
        v_full  = v_scalar[frozenset(view_names)]

        for m in view_names:
            if m in fixed_views:
                rows.append({
                    "fold_id": fold_id, "modality": m, "phi": phi[m],
                    "v_single": v_empty, "v_complement": v_full,
                    "marginal_grand": np.nan, "v_full": v_full,
                    "v_empty": v_empty, "n_free": n_free,
                })
                continue

            others   = [v for v in free_views if v != m]
            S_single = frozenset([m]) | fixed_set
            S_compl  = frozenset(others) | fixed_set  # N \ {m}

            v_single = v_scalar[S_single]
            v_compl  = v_scalar[S_compl]
            marginal_grand = v_full - v_compl

            rows.append({
                "fold_id": fold_id, "modality": m, "phi": phi[m],
                "v_single": v_single, "v_complement": v_compl,
                "marginal_grand": marginal_grand, "v_full": v_full,
                "v_empty": v_empty, "n_free": n_free,
            })

    return pd.DataFrame(rows)


def add_phi_decomposition(df_shapley: pd.DataFrame) -> pd.DataFrame:
    """Add phi_grand = marginal_grand / n_free and phi_small = phi - phi_grand."""
    df = df_shapley.copy()
    df["phi_grand"] = df["marginal_grand"] / df["n_free"]
    df["phi_small"] = df["phi"] - df["phi_grand"]
    return df


def compute_redundancy_fold(v_dict: dict, free_views: list, metric: str,
                             fixed_views: list = None) -> pd.DataFrame:
    """
    RI(i,j) = v({i,j}) - v({i}) - v({j}) + v(emptyset), for one fold.
    Returns a long-form DataFrame with columns view_i, view_j, redundancy.
    """
    fixed_views = fixed_views or []
    fixed_set   = frozenset(fixed_views)
    v_scalar    = scalar_vdict(v_dict, metric)
    v_empty     = v_scalar.get(fixed_set, v_scalar.get(frozenset(), 0.0))

    rows = []
    for i, j in combinations(free_views, 2):
        v_ij = v_scalar[frozenset([i, j]) | fixed_set]
        v_i  = v_scalar[frozenset([i]) | fixed_set]
        v_j  = v_scalar[frozenset([j]) | fixed_set]
        rows.append({"view_i": i, "view_j": j,
                     "redundancy": v_ij - v_i - v_j + v_empty})
    return pd.DataFrame(rows)


def compute_redundancy_matrix(v_dicts: dict, free_views: list, metric: str,
                               fixed_views: list = None) -> tuple:
    """Mean/std redundancy matrices (DataFrames indexed/labeled by view) across folds."""
    n = len(free_views)
    all_vals = np.full((len(v_dicts), n, n), np.nan)

    for f_idx, (fold_id, v_dict) in enumerate(v_dicts.items()):
        df = compute_redundancy_fold(v_dict, free_views, metric, fixed_views=fixed_views)
        for _, row in df.iterrows():
            i = free_views.index(row["view_i"])
            j = free_views.index(row["view_j"])
            all_vals[f_idx, i, j] = row["redundancy"]
            all_vals[f_idx, j, i] = row["redundancy"]

    for i in range(n):
        all_vals[:, i, i] = 0.0
    mean_mat = pd.DataFrame(np.nanmean(all_vals, axis=0), index=free_views, columns=free_views)
    std_mat  = pd.DataFrame(np.nanstd(all_vals, axis=0), index=free_views, columns=free_views)
    return mean_mat, std_mat


def correlation_analysis(df_shapley: pd.DataFrame, df_ps: pd.DataFrame,
                          free_views: list) -> dict:
    """
    For each fold: Spearman rank corr, Pearson corr (phi vs PS_raw), pairwise
    ordering agreement, and the marginal_grand ~ PS_raw residual.

    df_shapley: output of build_shapley_table (has fold_id, modality, phi, marginal_grand)
    df_ps:      perceptual_score CSVs concatenated (has fold_id, modality, PS_raw)
    """
    merged = df_shapley[df_shapley["modality"].isin(free_views)].merge(
        df_ps[["fold_id", "modality", "PS_raw"]], on=["fold_id", "modality"]
    )

    per_fold = []
    for fold_id, grp in merged.groupby("fold_id"):
        grp = grp.set_index("modality").loc[free_views]
        spearman = stats.spearmanr(grp["phi"], grp["PS_raw"]).correlation
        pearson  = stats.pearsonr(grp["phi"], grp["PS_raw"])[0]

        n_agree, n_pairs = 0, 0
        for i, j in combinations(free_views, 2):
            phi_order = grp.loc[i, "phi"] - grp.loc[j, "phi"]
            ps_order  = grp.loc[i, "PS_raw"] - grp.loc[j, "PS_raw"]
            n_pairs += 1
            if np.sign(phi_order) == np.sign(ps_order):
                n_agree += 1

        residual = (grp["marginal_grand"] - grp["PS_raw"])
        per_fold.append({
            "fold_id": fold_id, "spearman": spearman, "pearson": pearson,
            "pairwise_agreement": n_agree / n_pairs,
            "residual_mean": residual.mean(), "residual_std": residual.std(),
            "residual_abs_mean": residual.abs().mean(),
        })

    df_corr = pd.DataFrame(per_fold)

    overall_spearman = stats.spearmanr(merged["phi"], merged["PS_raw"]).correlation
    overall_pearson  = stats.pearsonr(merged["phi"], merged["PS_raw"])[0]

    return {
        "per_fold": df_corr,
        "overall_spearman": overall_spearman,
        "overall_pearson": overall_pearson,
        "merged": merged,
    }
