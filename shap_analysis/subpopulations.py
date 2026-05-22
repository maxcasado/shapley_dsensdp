"""
shap_analysis/subpopulations.py
--------------------------------
Discover latent subpopulations in the dataset by clustering points according
to their signed Shapley profile (φ_S2, φ_S1, φ_weather, φ_DEM).

Unlike regime analysis (which normalises φ to capture *which* sensor dominates),
subpopulation analysis uses signed raw φ values to capture both the direction
and magnitude of each modality's contribution — revealing groups of points that
the model handles in fundamentally different ways.

----------
find_optimal_k_gmm  : BIC + silhouette selection for number of components.
compute_subpopulations : fit GMM and annotate df with subpopulation labels.
get_subpop_profiles    : mean ± std φ per subpopulation.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Optimal k selection ───────────────────────────────────────────────────────

def find_optimal_k_gmm(
    phi_matrix: np.ndarray,
    k_range: range = range(2, 9),
    n_init: int = 5,
    random_state: int = 0,
    covariance_type: str = "full",
) -> dict:
    """
    Select the number of GMM components via BIC and silhouette score.

    BIC rewards fit while penalising model complexity — lower is better.
    Silhouette measures cluster separation — higher is better.
    The recommended k balances both.

    Args:
        phi_matrix:       (n_points, n_views) signed Shapley values.
        k_range:          Range of k values to test.
        n_init:           GMM restarts per k (increases robustness).
        random_state:     Seed.
        covariance_type:  GMM covariance structure ("full", "diag", "tied").

    Returns:
        Dict with keys: k_values, bics, silhouettes, best_k_bic,
                        best_k_sil, recommended_k.
    """
    X = StandardScaler().fit_transform(phi_matrix)

    bics        = []
    silhouettes = []

    for k in k_range:
        gmm    = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                 n_init=n_init, random_state=random_state)
        labels = gmm.fit_predict(X)
        bics.append(gmm.bic(X))
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        silhouettes.append(sil)
        print(f"  k={k}  BIC={bics[-1]:.0f}  silhouette={sil:.4f}", flush=True)

    best_k_bic = list(k_range)[int(np.argmin(bics))]
    best_k_sil = list(k_range)[int(np.argmax(silhouettes))]

    # Recommended: prefer BIC but note if silhouette disagrees strongly
    recommended_k = best_k_bic
    if best_k_sil != best_k_bic:
        print(f"  Note: BIC recommends k={best_k_bic}, "
              f"silhouette recommends k={best_k_sil}. "
              f"Using BIC (k={best_k_bic}).", flush=True)

    print(f"  → Recommended k: {recommended_k}", flush=True)

    return {
        "k_values":      list(k_range),
        "bics":          bics,
        "silhouettes":   silhouettes,
        "best_k_bic":    best_k_bic,
        "best_k_sil":    best_k_sil,
        "recommended_k": recommended_k,
    }


# ── Subpopulation computation ─────────────────────────────────────────────────

def compute_subpopulations(
    df: pd.DataFrame,
    view_names: list,
    k: int,
    n_init: int = 10,
    random_state: int = 0,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """
    Fit a GMM on signed Shapley profiles and annotate df with subpopulation labels.

    The GMM is fitted on StandardScaler-normalised φ values so that all
    modalities contribute equally to the clustering regardless of their
    absolute magnitude.

    Args:
        df:         DataFrame with <view>_shapley columns.
        view_names: List of modality names.
        k:          Number of Gaussian components.
        n_init:     GMM restarts for robustness.
        random_state: Seed.
        covariance_type: "full" (default) or "diag".

    Returns:
        df with new columns:
            "subpop"       : integer label 0…k-1
            "subpop_label" : human-readable label (e.g. "P1 — S2↑ DEM↓")
            "subpop_prob"  : GMM posterior probability of assigned component
    """
    phi_cols   = [f"{v}_shapley" for v in view_names]
    phi_matrix = df[phi_cols].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(phi_matrix)

    gmm    = GaussianMixture(n_components=k, covariance_type=covariance_type,
                             n_init=n_init, random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs  = gmm.predict_proba(X).max(axis=1)

    df = df.copy()
    df["subpop"]      = labels
    df["subpop_prob"] = probs

    # ── Build labels based on mean signed φ ───────────────────────────────────
    profiles = get_subpop_profiles(df, view_names)
    subpop_labels = {}

    for sp_id, row in profiles.iterrows():
        n      = int(row["n_points"])
        means  = np.array([row[f"mean_phi_{v}"] for v in view_names])
        order  = np.argsort(np.abs(means))[::-1]   # most influential first

        # Show the top 2 most influential with sign
        parts = []
        for idx in order[:2]:
            view = view_names[idx]
            sign = "↑" if means[idx] > 0 else "↓"
            parts.append(f"{view}{sign}")

        subpop_labels[sp_id] = f"P{sp_id+1} — {' '.join(parts)}  (n={n})"

    df["subpop_label"] = df["subpop"].map(subpop_labels)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}\n  Subpopulations (GMM k={k})\n{sep}")
    for sp_id, row in profiles.iterrows():
        print(f"\n  {subpop_labels[sp_id]}")
        print(f"  {'─'*44}")
        for view in view_names:
            mean = row[f"mean_phi_{view}"]
            std  = row[f"std_phi_{view}"]
            sign = "+" if mean >= 0 else ""
            bar_len = int(abs(mean) * 30)
            bar = ("+" if mean >= 0 else "-") * bar_len
            print(f"    {view:<12}  {bar:<30}  {sign}{mean:.3f} ± {std:.3f}")
    print(sep)

    return df


# ── Profile extraction ────────────────────────────────────────────────────────

def get_subpop_profiles(df: pd.DataFrame, view_names: list) -> pd.DataFrame:
    """
    Compute mean ± std of signed φ per view per subpopulation.

    Args:
        df:         DataFrame with 'subpop' column (from compute_subpopulations).
        view_names: List of modality names.

    Returns:
        DataFrame indexed by subpop_id with columns mean_phi_{view},
        std_phi_{view}, n_points, and optional crop_rate if 'label' is present.
    """
    phi_cols = [f"{v}_shapley" for v in view_names]
    k        = df["subpop"].nunique()
    rows     = []

    for sp_id in range(k):
        sub = df[df["subpop"] == sp_id]
        row = {"subpop_id": sp_id, "n_points": len(sub)}
        for col, view in zip(phi_cols, view_names):
            row[f"mean_phi_{view}"] = sub[col].mean()
            row[f"std_phi_{view}"]  = sub[col].std()
        if "label" in df.columns:
            row["crop_rate"] = sub["label"].mean()
        if "correct" in df.columns:
            row["accuracy"] = sub["correct"].mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index("subpop_id")