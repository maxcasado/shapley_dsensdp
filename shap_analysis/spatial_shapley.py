"""
shap_analysis/spatial_shapley.py
---------------------------------
Per-point spatial Shapley values and Shapley Interaction Index (SII).

Unlike the metric-based Shapley in shapley.py, the characteristic function
here is the raw model logit at each individual sample, enabling spatial maps
of sensor contributions.

Public API
----------
compute_spatial_shapley      : run all subset inferences, return per-point
                               Shapley matrix + full predictions + v_dict.
compute_shapley_interactions : SII order-2 for all view pairs.
shapley_bbox_per_point       : high-level wrapper → filtered DataFrame.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from code.datasets.utils import create_dataloader


# ── Low-level inference ───────────────────────────────────────────────────────

def _infer_subset(method, data_te, subset_list: list, batch_size: int) -> np.ndarray:
    """
    Run inference using only the views in subset_list.

    Returns raw logits of shape (n_samples,) or (n_samples, n_classes).
    """
    with torch.no_grad():
        out = method.transform(
            create_dataloader(data_te, batch_size=batch_size, train=False),
            out_norm=None,
            args_forward={
                "inference_views": subset_list,
                "missing_method":  method.missing_method,
            },
            perc_forward=1.0,
            not_return_repre=True,
        )
    return out["prediction"]


# ── Core spatial Shapley ──────────────────────────────────────────────────────

def compute_spatial_shapley(
    method,
    data_te,
    bbox_idx: np.ndarray,
    batch_size: int,
    task_type: str,
    view_names: list,
    baseline: float = None,
) -> tuple:
    """
    Compute per-point Shapley values for samples in bbox_idx.

    The characteristic function v(S, x) is the model logit at point x when
    only the views in S are observed. For multiclass tasks, the logit of the
    class predicted by the full model is used as reference.

    Args:
        method:     Trained model (eval mode, missing_info already set).
        data_te:    Full test Dataset_MultiView.
        bbox_idx:   Indices of the samples to explain (bounding box filter).
        batch_size: Inference batch size.
        task_type:  "classification" | "multilabel".
        view_names: Full list of sensor/modality names.
        baseline:   Value for v(∅, x) applied to all points.
                    If None, uses the proportion of positive labels in data_te
                    (task-specific crop baseline). Pass 0.0 for a neutral baseline.

    Returns:
        (shapley_matrix, full_preds, v_dict)
        - shapley_matrix : np.ndarray (n_bbox, n_views)
        - full_preds     : np.ndarray (n_bbox,) — logits with all views
        - v_dict         : {frozenset → np.ndarray (n_bbox,)} for all coalitions
    """
    n_views = len(view_names)
    n_bbox  = len(bbox_idx)

    # ── Baseline v(∅) ─────────────────────────────────────────────────────────
    if baseline is None:
        labels  = data_te.get_all_labels()
        baseline = float(np.mean(labels == 1))
        print(f"  Baseline v(∅) = {baseline:.4f} (positive class proportion)", flush=True)

    v_dict    = {frozenset(): np.full(n_bbox, baseline, dtype=np.float64)}
    ref_class = None  # fixed on the full-view pass (multiclass)

    # ── All subsets, size 1 → N ───────────────────────────────────────────────
    all_subsets   = [list(c) for s in range(1, n_views + 1)
                     for c in itertools.combinations(view_names, s)]
    n_subsets = len(all_subsets)
    print(f"  Running {n_subsets} inference passes (2^{n_views}-1 subsets)...", flush=True)

    for i, subset_list in enumerate(all_subsets):
        label       = "+".join(subset_list)
        logits_all  = _infer_subset(method, data_te, subset_list, batch_size)
        logits_bbox = logits_all[bbox_idx]

        # Multiclass: project onto reference class from the full-view prediction
        if logits_bbox.ndim > 1:
            fs = frozenset(subset_list)
            if fs == frozenset(view_names):
                ref_class = logits_bbox.argmax(axis=-1)        # (n_bbox,)
            if ref_class is not None:
                logits_bbox = logits_bbox[np.arange(n_bbox), ref_class]
            else:
                logits_bbox = logits_bbox.max(axis=-1)         # temporary proxy

        v_dict[frozenset(subset_list)] = logits_bbox.astype(np.float64)
        print(f"    [{i+1}/{n_subsets}] {label}", flush=True)

    # ── Per-point Shapley values ──────────────────────────────────────────────
    print("  Computing per-point Shapley values...", flush=True)
    shapley_matrix = np.zeros((n_bbox, n_views), dtype=np.float64)

    for j, view in enumerate(view_names):
        others = [v for v in view_names if v != view]
        phi    = np.zeros(n_bbox, dtype=np.float64)
        for size in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, size):
                S      = frozenset(S_tuple)
                s      = len(S)
                weight = (math.factorial(s) * math.factorial(n_views - s - 1)
                          / math.factorial(n_views))
                phi   += weight * (v_dict[S | {view}] - v_dict[S])
        shapley_matrix[:, j] = phi

    full_preds = v_dict[frozenset(view_names)]
    return shapley_matrix, full_preds, v_dict


# ── Shapley Interaction Index (SII) order 2 ───────────────────────────────────

def compute_shapley_interactions(
    v_dict: dict,
    view_names: list,
    n_bbox: int,
) -> np.ndarray:
    """
    Compute the Shapley Interaction Index (SII) order 2 for all view pairs.

    SII(i,j) = Σ_{S ⊆ N\{i,j}} w(|S|) · Δ_{ij}(S)
    Δ_{ij}(S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
    w(s)      = s! · (n-s-2)! / (n-1)!

    Reuses the v_dict already built by compute_spatial_shapley.

    Args:
        v_dict:     {frozenset → np.ndarray (n_bbox,)} from compute_spatial_shapley.
        view_names: Full list of view names.
        n_bbox:     Number of points (bbox size).

    Returns:
        interaction_matrix: np.ndarray (n_bbox, n_views, n_views), symmetric.
    """
    n_views = len(view_names)
    interaction_matrix = np.zeros((n_bbox, n_views, n_views), dtype=np.float64)

    for i, vi in enumerate(view_names):
        for j, vj in enumerate(view_names):
            if j <= i:
                continue
            others = [v for v in view_names if v not in (vi, vj)]
            phi_ij = np.zeros(n_bbox, dtype=np.float64)
            for size in range(len(others) + 1):
                for S_tuple in itertools.combinations(others, size):
                    S      = frozenset(S_tuple)
                    s      = len(S)
                    weight = (math.factorial(s) * math.factorial(n_views - s - 2)
                              / math.factorial(n_views - 1))
                    delta  = (v_dict[S | {vi, vj}]
                              - v_dict[S | {vi}]
                              - v_dict[S | {vj}]
                              + v_dict[S])
                    phi_ij += weight * delta
            interaction_matrix[:, i, j] = phi_ij
            interaction_matrix[:, j, i] = phi_ij   # symmetric

    print("  SII computed for all pairs.", flush=True)
    return interaction_matrix


# ── High-level wrapper ────────────────────────────────────────────────────────

def shapley_bbox_per_point(
    method,
    data_te,
    view_names: list,
    coords: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    batch_size: int = 32,
    baseline: float = 0.0,
    out_csv: str = None,
) -> pd.DataFrame:
    """
    Compute per-point Shapley values for samples within a bounding box.

    High-level wrapper around compute_spatial_shapley that handles filtering,
    result assembly, and optional CSV export.

    Args:
        method:     Trained model (eval mode).
        data_te:    Full test Dataset_MultiView.
        view_names: Full list of sensor/modality names.
        coords:     (n_samples, 2) array [longitude, latitude], aligned with data_te.
        lon/lat min/max: Bounding box extent.
        batch_size: Inference batch size.
        baseline:   v(∅) scalar applied uniformly (default 0.0).
        out_csv:    Optional path to save the result CSV.

    Returns:
        DataFrame with columns: sample_idx, lon, lat, <view>_shapley per view.
    """
    n_samples = len(data_te)
    if len(coords) != n_samples:
        raise ValueError(f"coords has {len(coords)} rows but data_te has {n_samples} samples.")

    lons = coords[:, 0]
    lats = coords[:, 1]
    mask     = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    bbox_idx = np.where(mask)[0]
    n_bbox   = len(bbox_idx)

    if n_bbox == 0:
        print("No points found in the bounding box.", flush=True)
        return pd.DataFrame()

    print(f"{n_bbox} points in bounding box (out of {n_samples} total).", flush=True)

    shapley_matrix, _, _ = compute_spatial_shapley(
        method, data_te, bbox_idx, batch_size,
        task_type=None, view_names=view_names, baseline=baseline,
    )

    df_dict = {"sample_idx": bbox_idx, "lon": lons[bbox_idx], "lat": lats[bbox_idx]}
    for j, view in enumerate(view_names):
        df_dict[f"{view}_shapley"] = shapley_matrix[:, j]
    df = pd.DataFrame(df_dict)

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Results saved -> {out_path}", flush=True)

    sep = "-" * 48
    print(f"\n{sep}")
    print(f"  Mean Shapley over bounding box ({n_bbox} points)")
    print(sep)
    for view in view_names:
        col  = f"{view}_shapley"
        mean = df[col].mean()
        std  = df[col].std()
        sign = "+" if mean >= 0 else ""
        print(f"  {view:<20} {sign}{mean:.4f}  (+/- {std:.4f})")
    print(sep)

    return df