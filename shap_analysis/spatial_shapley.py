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
    fixed_views: list = [],
) -> tuple:
    """
    Compute per-point Shapley values for samples in bbox_idx.

    Args:
        fixed_views: Views always present in every coalition — they are not
                     explained (no φ computed for them) but always passed to
                     the model. Useful for geo-coordinates that are always
                     available in practice.
    """
    n_views = len(view_names)
    n_bbox  = len(bbox_idx)

    # Split free vs fixed views
    free_views = [v for v in view_names if v not in fixed_views]
    fixed_set  = [v for v in view_names if v in fixed_views]
    n_free     = len(free_views)

    if fixed_set:
        print(f"  Fixed views (always in coalition): {fixed_set}", flush=True)
        print(f"  Free views (explained): {free_views}", flush=True)

    # ── Baseline v(∅) ─────────────────────────────────────────────────────────
    if baseline is None:
        labels   = data_te.get_all_labels()
        baseline = float(np.mean(labels == 1))
        print(f"  Baseline v(∅) = {baseline:.4f} (positive class proportion)",
              flush=True)
    baseline_arr = np.full(n_bbox, baseline, dtype=np.float64)

    v_dict    = {frozenset(): baseline_arr}
    ref_class = None

    # Map fixed-only coalition to the baseline
    if fixed_set:
        v_dict[frozenset(fixed_set)] = baseline_arr

    # ── All subsets over FREE views only, fixed views always added ────────────
    all_subsets = [list(c) + fixed_set
                   for s in range(1, n_free + 1)
                   for c in itertools.combinations(free_views, s)]

    n_subsets = len(all_subsets)
    print(f"  Running {n_subsets} inference passes "
          f"(2^{n_free}-1 free subsets, fixed={fixed_set or 'none'})...", flush=True)

    for i, subset_list in enumerate(all_subsets):
        label       = "+".join(subset_list)
        logits_all  = _infer_subset(method, data_te, subset_list, batch_size)
        logits_bbox = logits_all[bbox_idx]

        if logits_bbox.ndim > 1:
            n_classes = logits_bbox.shape[1]
            if n_classes == 2:
                # Binary classification: apply softmax then extract P(crop=1)
                # so that v(S, x) ∈ [0,1] and pred_score >= 0.5 matches
                # the model's decision boundary directly.
                e = np.exp(logits_bbox - logits_bbox.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)
                logits_bbox = probs[:, 1]  # P(crop)
            else:
                # Multiclass: project onto the class predicted by the full model
                fs = frozenset(subset_list)
                if fs == frozenset(view_names):
                    ref_class = logits_bbox.argmax(axis=-1)    # (n_bbox,)
                if ref_class is not None:
                    logits_bbox = logits_bbox[np.arange(n_bbox), ref_class]
                else:
                    logits_bbox = logits_bbox.max(axis=-1)     # temporary proxy

        v_dict[frozenset(subset_list)] = logits_bbox.astype(np.float64)
        print(f"    [{i+1}/{n_subsets}] {label}", flush=True)

    # ── Per-point Shapley values (free views only) ────────────────────────────
    print("  Computing per-point Shapley values...", flush=True)
    shapley_matrix = np.zeros((n_bbox, n_views), dtype=np.float64)

    for j, view in enumerate(view_names):
        if view in fixed_set:
            continue  # no Shapley value for fixed views
        others = [v for v in free_views if v != view]
        phi    = np.zeros(n_bbox, dtype=np.float64)
        for size in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, size):
                S      = frozenset(S_tuple) | frozenset(fixed_set)
                s      = len(frozenset(S_tuple))   # size without fixed
                weight = (math.factorial(s) * math.factorial(n_free - s - 1)
                          / math.factorial(n_free))
                phi   += weight * (v_dict[S | {view}] - v_dict[S])
        shapley_matrix[:, j] = phi

    full_preds = v_dict[frozenset(view_names)]
    return shapley_matrix, full_preds, v_dict


# ── Shapley Interaction Index (SII) order 2 ───────────────────────────────────

def compute_shapley_interactions(
    v_dict: dict,
    view_names: list,
    n_bbox: int,
    fixed_views: list = [],
) -> np.ndarray:
    """
    Compute SII order-2 for all view pairs. Fixed views are always included
    in every coalition and skipped in the interaction computation.
    """
    n_views = len(view_names)
    interaction_matrix = np.zeros((n_bbox, n_views, n_views), dtype=np.float64)

    # Only keep fixed views that are actually present in v_dict keys
    all_in_keys = set().union(*v_dict.keys()) if v_dict else set()
    fixed_set   = frozenset(v for v in fixed_views if v in all_in_keys)
    free_views  = [v for v in view_names
                   if v not in fixed_views and v in all_in_keys]
    n_free = len(free_views)

    for i, vi in enumerate(view_names):
        for j, vj in enumerate(view_names):
            if j <= i:
                continue
            if vi in fixed_views or vj in fixed_views:
                continue
            if vi not in free_views or vj not in free_views:
                continue  # skip views absent from v_dict
            others = [v for v in free_views if v not in (vi, vj)]
            phi_ij = np.zeros(n_bbox, dtype=np.float64)
            for size in range(len(others) + 1):
                for S_tuple in itertools.combinations(others, size):
                    S      = frozenset(S_tuple) | fixed_set
                    s      = len(frozenset(S_tuple))
                    weight = (math.factorial(s) * math.factorial(n_free - s - 2)
                              / math.factorial(n_free - 1))
                    delta  = (v_dict[S | {vi, vj}]
                              - v_dict[S | {vi}]
                              - v_dict[S | {vj}]
                              + v_dict[S])
                    phi_ij += weight * delta
            interaction_matrix[:, i, j] = phi_ij
            interaction_matrix[:, j, i] = phi_ij

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