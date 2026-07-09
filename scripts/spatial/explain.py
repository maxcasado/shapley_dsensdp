"""
explain.py
----------
For each test set point within a geographic bounding box, compute per-point
Shapley values for each modality and generate spatial maps and grouped statistics.

Usage:
    # Single fold
    python -m scripts.spatial.explain -s config/dsensdp_ex.yaml -r 0 -f 0

    # All folds (full k-fold coverage — recommended)
    python -m scripts.spatial.explain -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4

    # Geographic subset
    python -m scripts.spatial.explain -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 \
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
    from utils.geo import assign_continents, assign_climate_zones
    from shap_analysis.spatial_shapley import compute_spatial_shapley, compute_shapley_interactions
    from shap_analysis.regimes import find_optimal_k, compute_regimes
    from shap_analysis.subpopulations import find_optimal_k_gmm, compute_subpopulations
    from visualize.maps import (
        plot_shapley_maps,
        plot_gt_pred_maps,
        plot_continent_map,
        plot_interaction_graph,
        plot_interaction_maps,
        plot_interaction_matrix,
        compute_stats,
        plot_dominant_modality_by_continent,
        plot_regime_map,
        plot_regime_profiles,
        plot_regime_continent,
        plot_regime_silhouette,
        plot_tropical_comparison,
        plot_sii_by_climate_zone,
        plot_phi_vs_performance,
        plot_subpopulation_profiles,
        plot_subpop_k_selection,
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
    n_classes = len(np.unique(data_te.get_all_labels()))
    baseline  = 1.0 / n_classes  # uniform prior over classes
    shapley_matrix, full_preds, v_dict = compute_spatial_shapley(
        method, data_te, bbox_idx, batch_size, task_type, view_names,
        fixed_views=args.fixed_views,
        baseline=baseline,
    )
    log(f"  Shapley done in {time.time() - t0:.1f}s")

    # ── Interaction index ─────────────────────────────────────────────────────
    # Debug: print v_dict keys
    log(f"  v_dict keys ({len(v_dict)}): {sorted([str(sorted(k)) for k in list(v_dict.keys())[:8]])}")
    interaction_matrix = compute_shapley_interactions(
        v_dict, view_names, n_bbox, fixed_views=args.fixed_views)

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

    # Store predicted class for multiclass models
    # full_preds is P(argmax class) → need argmax over all classes
    full_logits = v_dict.get(frozenset(view_names))
    if full_logits is not None:
        from code.datasets.utils import create_dataloader
        with torch.no_grad():
            out_full = method.transform(
                create_dataloader(data_te, batch_size=batch_size, train=False),
                out_norm="softmax", not_return_repre=True,
            )
        probs_full = out_full["prediction"][bbox_idx]
        if probs_full.ndim > 1 and probs_full.shape[1] > 2:
            df_dict["pred_class"] = probs_full.argmax(axis=1)
        elif probs_full.ndim > 1 and probs_full.shape[1] == 2:
            df_dict["pred_class"] = (probs_full[:, 1] > 0.5).astype(int)
        else:
            df_dict["pred_class"] = (probs_full > 0.5).astype(int)
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
    df["continent"]    = assign_continents(df["lon"].values, df["lat"].values)
    df["climate_zone"] = assign_climate_zones(df["lat"].values)

    if "label" in df.columns and "pred_score" in df.columns:
        # For multiclass, pred_score is stored as the class index predicted
        # (argmax). Compare directly with label.
        if "pred_class" in df.columns:
            df["correct"] = df["pred_class"].values == df["label"].values.astype(int)
        else:
            # Binary fallback: pred_score > 0.5 (probability) or > 0 (log-odds)
            threshold = 0.5 if df["pred_score"].max() <= 1.0 else 0.0
            pred_class = (df["pred_score"].values > threshold).astype(int)
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

    if args.auto_zoom:
        pad = 5.0
        bbox_kw = dict(
            lon_min=float(df["lon"].min()) - pad,
            lon_max=float(df["lon"].max()) + pad,
            lat_min=float(df["lat"].min()) - pad,
            lat_max=float(df["lat"].max()) + pad,
        )
        log(f"Auto-zoom: {bbox_kw}")
    else:
        bbox_kw = dict(lon_min=args.lon_min, lon_max=args.lon_max,
                       lat_min=args.lat_min, lat_max=args.lat_max)

    # ── Plots ─────────────────────────────────────────────────────────────────
    log("\nComputing grouped statistics...")
    compute_stats(df, view_names, out_dir=out_root)
    plot_continent_map(df, out_dir=out_root, **bbox_kw)
    plot_dominant_modality_by_continent(df, view_names, out_dir=out_root / "stats")

    log("\nAnalysing tropical vs non-tropical contributions...")
    plot_tropical_comparison(df, view_names,
                             out_dir=out_root / "tropical" / "abs",
                             signed=False, **bbox_kw)
    plot_tropical_comparison(df, view_names,
                             out_dir=out_root / "tropical" / "signed",
                             signed=True, **bbox_kw)

    log("\nComputing SII by climate zone...")
    plot_sii_by_climate_zone(df, shapley_matrix, interaction_matrix,
                             view_names, out_dir=out_root / "tropical" / "sii_by_zone")

    log("\nCorrelation φ vs local performance...")
    plot_phi_vs_performance(df, view_names,
                            out_dir=out_root / "performance")

    # ── Subpopulation discovery via GMM ──────────────────────────────────────
    if args.subpopulations:
        dir_subpop  = out_root / "subpopulations"
        phi_matrix  = df[[f"{v}_shapley" for v in view_names]].values

        if args.n_subpop:
            k = args.n_subpop
            log(f"\nComputing {k} subpopulations (--n_subpop).")
        else:
            log("\nSelecting optimal number of subpopulations (GMM BIC)...")
            k_analysis = find_optimal_k_gmm(phi_matrix, k_range=range(2, 9))
            plot_subpop_k_selection(k_analysis, out_dir=dir_subpop)
            k = k_analysis["recommended_k"]
            log(f"\nSelected k={k} by BIC.")

        log(f"\nFitting GMM with k={k}...")
        df = compute_subpopulations(df, view_names, k=k)
        df.to_csv(out_root / "shapley_spatial.csv", index=False)
        log("CSV updated with subpopulation labels.")

        plot_subpopulation_profiles(df, view_names, out_dir=dir_subpop)

    log("\nGenerating GT / prediction maps...")
    n_classes = int(df["label"].nunique()) if "label" in df.columns else 2
    if n_classes <= 2:
        plot_gt_pred_maps(df, out_dir=dir_gt_pred, **bbox_kw)
    else:
        log(f"  Skipping binary GT/pred maps (multiclass with {n_classes} classes).")

    log("\nGenerating Shapley maps...")
    plot_shapley_maps(df, view_names, out_dir=dir_shapley, **bbox_kw)

    # Regional maps — zoom on data-dense areas
    if args.auto_zoom:
        from sklearn.cluster import KMeans
        import matplotlib.patches as mpatches
        coords_arr = df[["lon", "lat"]].values
        n_regions  = min(6, len(df) // 200)
        if n_regions >= 2:
            log(f"\nGenerating {n_regions} regional Shapley maps...")
            km = KMeans(n_clusters=n_regions, random_state=0, n_init=10)
            km.fit(coords_arr)

            # ── World overview with cluster bounding boxes ────────────────
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            try:
                from visualize.maps import _get_world_borders, _draw_borders
                borders = _get_world_borders()
            except Exception:
                borders = None

            PALETTE = ["#e15759","#4e79a7","#f28e2b","#76b7b2",
                       "#59a14f","#edc948"]
            pad = 3.0

            fig, ax = plt.subplots(figsize=(14, 7))
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.set_title("Data clusters — regional Shapley analysis", fontsize=13)

            if borders is not None:
                try:
                    from visualize.maps import _draw_borders
                    _draw_borders(ax, borders, -180, 180, -90, 90)
                except Exception:
                    pass

            legend_patches = []
            region_boxes   = []

            for reg_id in range(n_regions):
                mask   = km.labels_ == reg_id
                df_reg = df[mask]
                color  = PALETTE[reg_id % len(PALETTE)]

                # Scatter points
                ax.scatter(df_reg["lon"].values, df_reg["lat"].values,
                           c=color, s=1.5, linewidths=0,
                           rasterized=True, alpha=0.6, zorder=2)

                # Bounding box
                x0 = float(df_reg["lon"].min()) - pad
                x1 = float(df_reg["lon"].max()) + pad
                y0 = float(df_reg["lat"].min()) - pad
                y1 = float(df_reg["lat"].max()) + pad
                rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                     linewidth=2, edgecolor=color,
                                     facecolor="none", zorder=3)
                ax.add_patch(rect)
                ax.text(x0 + (x1-x0)/2, y1 + 1.5, f"R{reg_id}",
                        ha="center", va="bottom", fontsize=10,
                        color=color, fontweight="bold")

                legend_patches.append(
                    mpatches.Patch(color=color,
                                   label=f"R{reg_id} (n={mask.sum()})"))
                region_boxes.append((reg_id, x0, x1, y0, y1))

            ax.legend(handles=legend_patches, loc="lower left",
                      fontsize=9, framealpha=0.85, title="Regions")
            ax.grid(linestyle="--", alpha=0.3, zorder=0)
            plt.tight_layout()
            p = dir_shapley / "regions_overview.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plt.close()
            log(f"  -> regions_overview.png")

            # ── Per-region zoomed maps ────────────────────────────────────
            for reg_id, x0, x1, y0, y1 in region_boxes:
                mask   = km.labels_ == reg_id
                df_reg = df[mask]
                reg_kw = dict(lon_min=x0, lon_max=x1,
                              lat_min=y0, lat_max=y1)
                plot_shapley_maps(df_reg, view_names,
                                  out_dir=dir_shapley / f"region_{reg_id}",
                                  resolution=0.2,
                                  **reg_kw)
                log(f"  -> region_{reg_id} ({mask.sum()} points)")

    if "pred_class" in df.columns and df["pred_class"].nunique() > 2:
        log("\nGenerating per-class Shapley maps...")
        from visualize.maps import plot_shapley_maps_by_class, plot_shapley_kde_by_class
        plot_shapley_maps_by_class(df, view_names,
                                   out_dir=dir_shapley / "by_class",
                                   **bbox_kw)
        log("\nGenerating signed φ KDE by class...")
        plot_shapley_kde_by_class(df, view_names,
                                  out_dir=dir_shapley / "by_class")

    log("\nGenerating interaction plots...")
    plot_interaction_graph(shapley_matrix, interaction_matrix, view_names, out_dir=dir_interact)
    plot_interaction_maps(df, interaction_matrix, view_names, out_dir=dir_interact, **bbox_kw)
    plot_interaction_matrix(shapley_matrix, interaction_matrix, view_names, out_dir=dir_interact)

    # ── Sensor regime analysis (optional) ────────────────────────────────────
    if args.regimes:
        dir_regimes = out_root / "regimes"
        phi_matrix  = df[[f"{v}_shapley" for v in view_names]].values

        if args.n_regimes:
            k = args.n_regimes
            log(f"\nUsing k={k} regimes (--n_regimes).")
        else:
            log("\nSelecting optimal number of regimes...")
            k_analysis = find_optimal_k(phi_matrix, k_range=range(2, min(9, len(df))))
            plot_regime_silhouette(k_analysis, out_dir=dir_regimes)
            k = k_analysis["best_k"]
            log(f"\nSelected k={k} by silhouette score.")
        df = compute_regimes(df, view_names, k=k)

        df.to_csv(csv_path, index=False)  # update CSV with regime column
        log(f"CSV updated with regime labels -> {csv_path}")

        plot_regime_map(df, out_dir=dir_regimes, **bbox_kw)
        plot_regime_profiles(df, view_names, out_dir=dir_regimes)
        plot_regime_continent(df, out_dir=dir_regimes)

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
    p.add_argument("--regimes", action="store_true",
                   help="Run sensor regime analysis (KMeans on Shapley profiles).")
    p.add_argument("--n_regimes", type=int, default=None,
                   help="Number of regimes (k). If not set, selected automatically "
                        "by silhouette score.")
    p.add_argument("--subpopulations", action="store_true",
                   help="Run subpopulation discovery (GMM on signed φ profiles).")
    p.add_argument("--n_subpop", type=int, default=None,
                   help="Number of subpopulations. If not set, selected by BIC.")
    p.add_argument("--fixed_views", nargs="*", default=[],
                   help="Views always included in every coalition (not explained). "
                        "e.g. --fixed_views geo")
    p.add_argument("--auto_zoom", action="store_true",
                   help="Auto-zoom maps to data extent instead of full world view.")

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