"""
explain.py
----------
For each test set point within a geographic bounding box, compute per-point
Shapley values for each modality and generate spatial maps and grouped statistics.

Usage:
    # Single fold
    python explain.py -s config/dsensdp_ex.yaml -r 0 -f 0

    # All folds (full k-fold coverage — recommended)
    python explain.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4

    # Geographic subset
    python explain.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 \
        --lon_min -5.5 --lon_max 9.5 --lat_min 41.0 --lat_max 51.5
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

print("Standard imports OK", flush=True)

try:
    from utils.checkpoint import build_model, load_test_data, resolve_weights_path
    from utils.geo import assign_continents
    from shap_analysis.spatial_shapley import compute_spatial_shapley, compute_shapley_interactions
    from visualize.maps import (
        plot_shapley_maps,
        plot_gt_pred_maps,
        plot_continent_map,
        plot_interaction_graph,
        plot_interaction_maps,
        plot_interaction_matrix,
        compute_stats,
    )
    print("Project imports OK", flush=True)
except Exception:
    print(f"\nImport error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


# Default bounding box covers the full world.
# Restrict with --lon_min/max --lat_min/max for a specific region.
DEFAULT_LON_MIN = -180
DEFAULT_LON_MAX =  180
DEFAULT_LAT_MIN =  -90
DEFAULT_LAT_MAX =  90
NC_PATH         = "/home/casado/DsensDp/data/out/cropharvest_binary.nc"


def log(msg: str):
    print(msg, flush=True)


# ==============================================================================
#  Coordinate loading (with caching for multi-fold runs)
# ==============================================================================

def _load_id2coord(nc_path: str) -> dict:
    """Load the full identifier → [lon, lat] mapping from the NetCDF file."""
    log(f"Loading coordinates from {nc_path}...")
    ds        = xr.open_dataset(nc_path)
    ids_nc    = ds["identifier"].values
    coords_nc = ds["coords"].values          # (n_total, 2) -> [lon, lat]
    log(f"  {len(ids_nc)} coordinates loaded.")
    return {id_: coords_nc[i] for i, id_ in enumerate(ids_nc)}


def _coords_for(data_te, id2coord: dict) -> np.ndarray:
    """Return (n_samples, 2) coordinate array aligned with data_te."""
    ids_te  = data_te.get_all_identifiers()
    missing = [id_ for id_ in ids_te if id_ not in id2coord]
    if missing:
        raise KeyError(f"{len(missing)} identifiers missing from .nc: {missing[:5]}")
    return np.array([id2coord[id_] for id_ in ids_te])


# ==============================================================================
#  Per-fold computation
# ==============================================================================

def _run_fold(
    run_id: int,
    fold_id: int,
    args,
    config: dict,
    id2coord: dict,
) -> tuple:
    """
    Compute spatial Shapley values for one fold's test set.

    Returns:
        df              : DataFrame with shapley columns + metadata
        shapley_matrix  : np.ndarray (n_bbox, n_views)
        interaction_matrix : np.ndarray (n_bbox, n_views, n_views)
        view_names      : list of str
    """
    log(f"\n{'─'*60}")
    log(f"  Run {run_id} / Fold {fold_id}")
    log(f"{'─'*60}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args.weights_file and len(args.fold_ids) == 1:
        weights_path = Path(args.weights_file)
    else:
        weights_path = resolve_weights_path(config, run_id, fold_id)

    if not weights_path.exists():
        log(f"  ERROR: checkpoint not found: {weights_path}")
        return None, None, None, None

    log(f"  Checkpoint: {weights_path}")
    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)

    method = build_model(checkpoint, ckpt_config)
    data_te = load_test_data(checkpoint, ckpt_config)

    # ── Coordinates & bounding box ────────────────────────────────────────────
    coords   = _coords_for(data_te, id2coord)
    lons     = coords[:, 0]
    lats     = coords[:, 1]
    mask     = ((lons >= args.lon_min) & (lons <= args.lon_max) &
                (lats >= args.lat_min) & (lats <= args.lat_max))
    bbox_idx = np.where(mask)[0]
    n_bbox   = len(bbox_idx)

    if n_bbox == 0:
        log("  No points in bounding box for this fold, skipping.")
        return None, None, None, None

    log(f"  {n_bbox} points in bounding box (out of {len(data_te)} total).")

    # ── Shapley computation ───────────────────────────────────────────────────
    batch_size = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type  = ckpt_config.get("task_type", "")
    view_names = ckpt_config["experiment"]["preprocess"]["view_names"]
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    t0 = time.time()
    shapley_matrix, full_preds, v_dict = compute_spatial_shapley(
        method, data_te, bbox_idx, batch_size, task_type, view_names,
    )
    log(f"  Shapley done in {time.time() - t0:.1f}s")

    # ── Interaction index ─────────────────────────────────────────────────────
    interaction_matrix = compute_shapley_interactions(v_dict, view_names, n_bbox)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    ids_bbox = data_te.get_all_identifiers()[bbox_idx]
    df_dict  = {
        "run_id":     run_id,
        "fold_id":    fold_id,
        "sample_idx": bbox_idx,
        "identifier": ids_bbox,
        "lon":        lons[bbox_idx],
        "lat":        lats[bbox_idx],
        "pred_score": full_preds,
    }
    for j, view in enumerate(view_names):
        df_dict[f"{view}_shapley"] = shapley_matrix[:, j]
    for i, vi in enumerate(view_names):
        for j, vj in enumerate(view_names):
            if j > i:
                df_dict[f"sii_{vi}_{vj}"] = interaction_matrix[:, i, j]
    try:
        labels = data_te.get_all_labels()[bbox_idx]
        df_dict["label"] = labels if labels.ndim == 1 else labels[:, 0]
    except Exception:
        pass

    df = pd.DataFrame(df_dict)
    return df, shapley_matrix, interaction_matrix, view_names


# ==============================================================================
#  Main
# ==============================================================================

def main(args):
    log(f"\nReading config: {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.safe_load(f)

    fold_ids = args.fold_ids  # list of ints, e.g. [0, 1, 2, 3, 4]
    run_id   = args.run_id

    # ── Load coordinates once (shared across all folds) ───────────────────────
    id2coord = _load_id2coord(args.nc_path)

    # ── Iterate over folds ────────────────────────────────────────────────────
    all_dfs                 = []
    all_shapley_matrices    = []
    all_interaction_matrices = []
    view_names              = None

    for fold_id in fold_ids:
        df, shap_mat, inter_mat, vn = _run_fold(run_id, fold_id, args, config, id2coord)
        if df is None:
            continue
        all_dfs.append(df)
        all_shapley_matrices.append(shap_mat)
        all_interaction_matrices.append(inter_mat)
        if view_names is None:
            view_names = vn

    if not all_dfs:
        log("\nNo points found in bounding box across any fold. Exiting.")
        sys.exit(0)

    # ── Concatenate across folds ──────────────────────────────────────────────
    df                 = pd.concat(all_dfs, ignore_index=True)
    shapley_matrix     = np.concatenate(all_shapley_matrices, axis=0)    # (n_total, n_views)
    interaction_matrix = np.concatenate(all_interaction_matrices, axis=0) # (n_total, n_views, n_views)

    n_folds = len(all_dfs)
    log(f"\n{'═'*60}")
    log(f"  Combined: {len(df)} points across {n_folds} fold(s)")
    log(f"{'═'*60}")

    # ── Enrichment ────────────────────────────────────────────────────────────
    log("\nAssigning continents...")
    df["continent"] = assign_continents(df["lon"].values, df["lat"].values)

    if "label" in df.columns and "pred_score" in df.columns:
        pred_class    = (df["pred_score"].values >= 0.5).astype(int)
        df["correct"] = pred_class == df["label"].values.astype(int)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "shapley_spatial.csv"
    df.to_csv(csv_path, index=False)
    log(f"\nCSV saved -> {csv_path}  ({len(df)} rows)")

    # ── Optional geospatial export ────────────────────────────────────────────
    if args.geo_format:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            gdf      = gpd.GeoDataFrame(
                df, geometry=[Point(lo, la) for lo, la in zip(df["lon"], df["lat"])],
                crs="EPSG:4326",
            )
            geo_path = out_root / f"shapley_spatial.{args.geo_format}"
            driver   = {"gpkg": "GPKG", "geojson": "GeoJSON"}[args.geo_format]
            gdf.to_file(geo_path, driver=driver)
            log(f"Geospatial export saved -> {geo_path}")
        except ImportError:
            log("WARNING: geopandas/shapely unavailable, geospatial export skipped.")

    # ── Output directories ────────────────────────────────────────────────────
    dir_gt_pred  = out_root / "maps_gt_pred"
    dir_shapley  = out_root / "maps_shapley"
    dir_interact = out_root / "maps_interactions"
    bbox_kw      = dict(lon_min=args.lon_min, lon_max=args.lon_max,
                        lat_min=args.lat_min, lat_max=args.lat_max)

    # ── Plots ─────────────────────────────────────────────────────────────────
    log("\nComputing grouped statistics...")
    compute_stats(df, view_names, out_dir=out_root)
    plot_continent_map(df, out_dir=out_root, **bbox_kw)

    log("\nGenerating GT / prediction maps...")
    plot_gt_pred_maps(df, out_dir=dir_gt_pred, **bbox_kw)

    log("\nGenerating Shapley maps...")
    plot_shapley_maps(df, view_names, out_dir=dir_shapley, **bbox_kw)

    log("\nGenerating interaction plots...")
    plot_interaction_graph(shapley_matrix, interaction_matrix, view_names, out_dir=dir_interact)
    plot_interaction_maps(df, interaction_matrix, view_names, out_dir=dir_interact, **bbox_kw)
    plot_interaction_matrix(shapley_matrix, interaction_matrix, view_names, out_dir=dir_interact)

    log(f"\nDone. Output structure:")
    log(f"  {out_root}/")
    log(f"  ├── shapley_spatial.csv  ({len(df)} points, {n_folds} fold(s))")
    log(f"  ├── maps_gt_pred/")
    log(f"  ├── maps_shapley/")
    log(f"  ├── maps_interactions/")
    log(f"  └── stats/")


# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Per-point spatial Shapley values.")
    p.add_argument("--settings_file", "-s", required=True,
                   help="YAML config file.")
    p.add_argument("--weights_file",  "-w", default=None,
                   help="Explicit .pt path (only used for single-fold runs).")
    p.add_argument("--run_id",  "-r", type=int, default=0)

    # Fold selection: either -f (single) or --fold_ids (one or more)
    fold_group = p.add_mutually_exclusive_group()
    fold_group.add_argument("--fold_id",  "-f", type=int, default=None,
                            help="Single fold to evaluate (default: 0).")
    fold_group.add_argument("--fold_ids", type=int, nargs="+", default=None,
                            metavar="FOLD",
                            help="One or more fold IDs to evaluate and concatenate "
                                 "(e.g. --fold_ids 0 1 2 3 4).")

    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--nc_path",    default=NC_PATH)
    p.add_argument("--out_dir",    "-o", default="preds/spatial",
                   help="Output directory (default: preds/spatial).")
    p.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    p.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--geo_format", choices=["gpkg", "geojson"], default=None)

    args = p.parse_args()

    # Resolve fold_ids: --fold_id and --fold_ids are mutually exclusive;
    # if neither is given, default to fold 0.
    if args.fold_ids is not None:
        pass  # already set
    elif args.fold_id is not None:
        args.fold_ids = [args.fold_id]
    else:
        args.fold_ids = [0]

    return args


if __name__ == "__main__":
    try:
        main(parse_args())
    except SystemExit:
        raise
    except Exception:
        log("\nUnhandled error:\n")
        log(traceback.format_exc())
        sys.exit(1)