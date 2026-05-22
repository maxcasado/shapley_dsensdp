"""
shap_analysis/regimes.py
------------------------
Sensor regime analysis: cluster points by their Shapley value profile
(φ_S1, φ_S2, φ_weather, φ_DEM) to identify regions where the model
consistently relies on different subsets of sensors.

The clustering is done in the normalized Shapley space:
    φ̃ᵢ = φᵢ / Σⱼ|φⱼ|  (relative contribution per point)
so that the clustering captures *which sensors dominate*, not the overall
magnitude of Shapley values.

Public API
----------
find_optimal_k      : silhouette + elbow analysis to suggest k.
compute_regimes     : fit KMeans and return cluster labels + profiles.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ── Optimal k selection ───────────────────────────────────────────────────────

def find_optimal_k(
    phi_matrix: np.ndarray,
    k_range: range = range(2, 8),
    n_init: int = 20,
    random_state: int = 0,
) -> dict:
    """
    Compute silhouette scores and inertia for a range of k values.

    Args:
        phi_matrix:   (n_points, n_views) array of Shapley values.
        k_range:      Range of k values to evaluate (default 2–7).
        n_init:       KMeans restarts per k.
        random_state: Reproducibility seed.

    Returns:
        Dict with keys:
            "k_values"    : list of k values tested
            "silhouettes" : silhouette score per k
            "inertias"    : inertia (within-cluster sum of squares) per k
            "best_k"      : k with highest silhouette score
    """
    X = _normalize_profiles(phi_matrix)

    silhouettes = []
    inertias    = []

    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)
        print(f"  k={k}  silhouette={silhouettes[-1]:.4f}  inertia={inertias[-1]:.1f}",
              flush=True)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"  → Best k by silhouette: {best_k}", flush=True)

    return {
        "k_values":    list(k_range),
        "silhouettes": silhouettes,
        "inertias":    inertias,
        "best_k":      best_k,
    }


# ── Regime computation ────────────────────────────────────────────────────────

def compute_regimes(
    df: pd.DataFrame,
    view_names: list,
    k: int,
    n_init: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Cluster points by their normalized Shapley profile and annotate df.

    Args:
        df:         DataFrame with <view>_shapley columns per view.
        view_names: List of modality names.
        k:          Number of clusters (sensor regimes).
        n_init:     KMeans restarts.
        random_state: Seed.

    Returns:
        df with two new columns:
            "regime"       : cluster label (0-indexed integer)
            "regime_label" : human-readable label e.g. "Regime 1 (DEM-driven)"
        Also prints a summary table of mean |φ| per regime.
    """
    phi_cols   = [f"{v}_shapley" for v in view_names]
    phi_matrix = df[phi_cols].values

    X      = _normalize_profiles(phi_matrix)
    km     = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)

    df = df.copy()
    df["regime"] = labels

    # ── Build human-readable labels ───────────────────────────────────────────
    # Label each regime by its dominant view (highest mean |φ̃|)
    profiles = _compute_profiles(df, view_names, k)
    regime_labels = {}
    used_names    = {}   # base_name → count, to detect duplicates

    for regime_id, row in profiles.iterrows():
        mean_cols = [c for c in row.index if c.startswith("mean_phi_")]
        mean_vals = row[mean_cols].values
        order     = np.argsort(mean_vals)[::-1]   # descending
        n         = int(row["n_points"])

        dom_view  = view_names[order[0]]
        base_name = dom_view

        # If this dominant view already used, append secondary modality
        if base_name in used_names:
            sec_view  = view_names[order[1]]
            base_name = f"{dom_view}+{sec_view}"

        used_names[dom_view] = used_names.get(dom_view, 0) + 1
        regime_labels[regime_id] = f"R{regime_id+1} — {base_name}  (n={n})"

    df["regime_label"] = df["regime"].map(regime_labels)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}\n  Sensor regimes (k={k})\n{sep}")
    for regime_id, row in profiles.iterrows():
        print(f"\n  {regime_labels[regime_id]}")
        print(f"  {'─'*40}")
        for view in view_names:
            mean = row[f"mean_phi_{view}"]
            std  = row[f"std_phi_{view}"]
            bar  = "█" * int(mean * 40)
            print(f"    {view:<12}  {bar:<40}  {mean:.3f} ± {std:.3f}")
    print(sep)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_profiles(phi_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each point's Shapley vector by its L1 norm (sum of |φᵢ|).
    Points with all-zero φ are left as zeros.
    """
    abs_phi = np.abs(phi_matrix)
    row_sum = abs_phi.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)   # avoid div by zero
    return abs_phi / row_sum


def _compute_profiles(df: pd.DataFrame, view_names: list, k: int) -> pd.DataFrame:
    """
    Compute mean and std of |φ| per view per regime cluster.
    Returns a DataFrame indexed by regime_id.
    """
    rows = []
    for regime_id in range(k):
        sub  = df[df["regime"] == regime_id]
        row  = {"regime_id": regime_id, "n_points": len(sub)}
        for view in view_names:
            vals = sub[f"{view}_shapley"].abs()
            row[f"mean_phi_{view}"] = vals.mean()
            row[f"std_phi_{view}"]  = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime_id")


def get_regime_profiles(df: pd.DataFrame, view_names: list) -> pd.DataFrame:
    """
    Public helper: compute regime profiles from a df that already has
    a 'regime' column (output of compute_regimes).
    """
    k = df["regime"].nunique()
    return _compute_profiles(df, view_names, k)