"""
compare_spatial_stats.py
------------------------
Compare local TV and local Moran's I maps between two models,
using a shared colorbar scale for each (view, statistic) pair.

For each view: produces side-by-side maps of model_a vs model_b
with identical colorbar limits.

Usage:
    python -m scripts.spatial.compare_stats \\
        --csv_a preds_com/spatial/shapley_spatial.csv \\
        --csv_b preds_com_geo/spatial/shapley_spatial.csv \\
        --name_a com --name_b com_geo \\
        --out results/compare_com_vs_com_geo \\
        --threshold_km 25 --grid_resolution 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a",   required=True)
    p.add_argument("--csv_b",   required=True)
    p.add_argument("--name_a",  required=True)
    p.add_argument("--name_b",  required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--threshold_km", type=float, default=25.0)
    p.add_argument("--grid_resolution", type=float, default=2.0)
    p.add_argument("--signed",  action="store_true")
    p.add_argument("--pred_crop", action="store_true",
                   help="Only include points where model predicts crop (pred_score > 0)")
    return p.parse_args()


# ── Helpers (same as compute_point_spatial_stats.py) ─────────────────────────

def build_inv25(lats, lons, threshold_km=25.0):
    return _build_radius(lats, lons, threshold_km, mode="inv25")


def build_knn(lats, lons, k=5):
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
    tree = BallTree(coords_rad, metric="haversine")
    _, idx = tree.query(coords_rad, k=k + 1)
    indices = [row[1:] for row in idx]
    weights = [np.ones(k, dtype=np.float32) / k for _ in range(len(lats))]
    return indices, weights


def _build_radius(lats, lons, threshold_km, mode):
    coords_rad    = np.column_stack([np.radians(lats), np.radians(lons)])
    tree          = BallTree(coords_rad, metric="haversine")
    threshold_rad = threshold_km / EARTH_RADIUS_KM
    idx_list      = tree.query_radius(coords_rad, r=threshold_rad,
                                      return_distance=True, sort_results=True)
    idx_arr, dist_arr = idx_list[0], idx_list[1]
    indices, weights  = [], []
    for i in range(len(lats)):
        mask   = idx_arr[i] != i
        nb_idx = idx_arr[i][mask].astype(int)
        d_km   = dist_arr[i][mask] * EARTH_RADIUS_KM
        if len(nb_idx) == 0:
            indices.append(np.array([], dtype=int))
            weights.append(np.array([], dtype=np.float32))
            continue
        w = (1.0 / np.maximum(d_km, 1e-6)).astype(np.float32)
        w_sum = w.sum()
        if w_sum == 0:
            indices.append(np.array([], dtype=int))
            weights.append(np.array([], dtype=np.float32))
            continue
        indices.append(nb_idx)
        weights.append(w / w_sum)
    return indices, weights


def local_tv(phi, indices, weights):
    n  = len(phi)
    tv = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        nb = indices[i]; w = weights[i]
        if len(nb) == 0: continue
        tv[i] = np.dot(w, np.abs(phi[i] - phi[nb]))
    return tv


def local_moran(phi, indices, weights):
    mu  = np.nanmean(phi)
    std = np.nanstd(phi)
    if std == 0:
        return np.zeros(len(phi), dtype=np.float32), 0.0
    z    = (phi - mu) / std
    n    = len(phi)
    lisa = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        nb = indices[i]; w = weights[i]
        if len(nb) == 0: continue
        lisa[i] = z[i] * np.dot(w, z[nb])
    return lisa.astype(np.float32), float(np.nanmean(lisa))


def project_to_grid(lons, lats, values, resolution):
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_edges = np.arange(-90,   90 + resolution, resolution)
    n_lon = len(lon_edges) - 1
    n_lat = len(lat_edges) - 1
    lon_idx = np.clip(np.digitize(lons, lon_edges) - 1, 0, n_lon - 1)
    lat_idx = np.clip(np.digitize(lats, lat_edges) - 1, 0, n_lat - 1)
    flat    = lon_idx * n_lat + lat_idx
    valid   = ~np.isnan(values)
    gs = np.bincount(flat[valid], weights=values[valid],
                     minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    gc = np.bincount(flat[valid], minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    grid = np.where(gc > 0, gs / gc, np.nan)
    lc   = (lon_edges[:-1] + lon_edges[1:]) / 2
    ltc  = (lat_edges[:-1] + lat_edges[1:]) / 2
    return grid, lc, ltc


# ── Side-by-side comparison map ───────────────────────────────────────────────

def plot_comparison(grid_a, grid_b, lc, ltc,
                    name_a, name_b, view, stat,
                    cmap, vmin, vmax, center,
                    vals_a, vals_b, out_path):
    """
    Two maps side by side with shared colorbar + two overlapping distributions.
    """
    color_a = "#e15759"   # red for model A
    color_b = "#4e79a7"   # blue for model B

    fig = plt.figure(figsize=(19, 5))
    gs  = fig.add_gridspec(1, 4, width_ratios=[8, 8, 0.4, 1.2], wspace=0.08)
    ax_a    = fig.add_subplot(gs[0])
    ax_b    = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])
    ax_hist = fig.add_subplot(gs[3])

    kw = dict(cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

    for ax, grid, name in [(ax_a, grid_a, name_a), (ax_b, grid_b, name_b)]:
        pc = ax.pcolormesh(lc, ltc, grid.T, **kw)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name} — {view} ({stat})", fontsize=11)
        ax.set_xlabel("Longitude")
    ax_a.set_ylabel("Latitude")
    ax_b.set_yticks([])

    # Shared colorbar
    cb = fig.colorbar(pc, cax=ax_cbar)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(stat, fontsize=9)

    # Two overlapping distributions
    n_bins = 40
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bw = bins[1] - bins[0]

    for vals, color, name in [
        (vals_a, color_a, name_a),
        (vals_b, color_b, name_b),
    ]:
        v = np.clip(vals[~np.isnan(vals)], vmin, vmax)
        counts, _ = np.histogram(v, bins=bins)
        counts_norm = counts / max(counts.max(), 1)
        ax_hist.barh(bin_centers, counts_norm, height=bw * 0.85,
                     color=color, alpha=0.55, edgecolor="none", label=name)

    ax_hist.set_ylim(vmin, vmax)
    ax_hist.set_xlim(0, 1.15)
    ax_hist.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax_hist.legend(fontsize=7, loc="upper right",
                   framealpha=0.7, handlelength=1)
    if center:
        ax_hist.axhline(0, color="#888888", linewidth=0.7, linestyle="--")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.grid_resolution

    # Load both CSVs
    print(f"Loading {args.csv_a}...", flush=True)
    df_a = pd.read_csv(args.csv_a)
    print(f"Loading {args.csv_b}...", flush=True)
    df_b = pd.read_csv(args.csv_b)

    # Common view names (intersection)
    views_a = [c.replace("_shapley", "") for c in df_a.columns
               if c.endswith("_shapley")]
    views_b = [c.replace("_shapley", "") for c in df_b.columns
               if c.endswith("_shapley")]
    view_names = [v for v in views_a if v in views_b]
    print(f"Common views: {view_names}\n")

    phi_label = "φ" if args.signed else "|φ|"

    if args.pred_crop:
        for name, df in [("A", df_a), ("B", df_b)]:
            if "pred_score" not in df.columns:
                print(f"ERROR: --pred_crop requires 'pred_score' in {name} CSV")
                return
        n_a = len(df_a)
        n_b = len(df_b)
        # Auto-detect threshold: log-odds (centered at 0) or probability (centered at 0.5)
        thresh_a = 0.5 if df_a["pred_score"].max() <= 1.0 else 0.0
        thresh_b = 0.5 if df_b["pred_score"].max() <= 1.0 else 0.0
        df_a = df_a[df_a["pred_score"] > thresh_a].reset_index(drop=True)
        df_b = df_b[df_b["pred_score"] > thresh_b].reset_index(drop=True)
        print(f"  Filtered to pred_crop: "
              f"{args.name_a} {len(df_a)}/{n_a} (thresh={thresh_a})  "
              f"{args.name_b} {len(df_b)}/{n_b} (thresh={thresh_b})")

    # Build weight matrices — inv25 for TV, knn5 for Moran's I
    print("Building inv25 weight matrix for A...", flush=True)
    W_inv_a = build_inv25(df_a["lat"].values, df_a["lon"].values, args.threshold_km)
    print("Building inv25 weight matrix for B...", flush=True)
    W_inv_b = build_inv25(df_b["lat"].values, df_b["lon"].values, args.threshold_km)
    print("Building knn5 weight matrix for A...", flush=True)
    W_knn_a = build_knn(df_a["lat"].values, df_a["lon"].values, k=5)
    print("Building knn5 weight matrix for B...", flush=True)
    W_knn_b = build_knn(df_b["lat"].values, df_b["lon"].values, k=5)

    summary = []

    for view in view_names:
        col   = f"{view}_shapley"
        phi_a = df_a[col].values if args.signed else df_a[col].abs().values
        phi_b = df_b[col].values if args.signed else df_b[col].abs().values

        print(f"\n── {view} ──────────────────────────────────────────")

        # TV with inv25, Moran with knn5
        tv_a  = local_tv(phi_a, *W_inv_a)
        tv_b  = local_tv(phi_b, *W_inv_b)
        li_a, I_a = local_moran(phi_a, *W_knn_a)
        li_b, I_b = local_moran(phi_b, *W_knn_b)

        mean_tv_a = float(np.nanmean(tv_a))
        mean_tv_b = float(np.nanmean(tv_b))
        print(f"  mean TV   {args.name_a}={mean_tv_a:.4f}  {args.name_b}={mean_tv_b:.4f}")
        print(f"  Moran's I {args.name_a}={I_a:.4f}   {args.name_b}={I_b:.4f}")

        summary += [
            {"model": args.name_a, "view": view, "weight": "inv25",
             "mean_tv": mean_tv_a, "mean_moran": I_a},
            {"model": args.name_b, "view": view, "weight": "inv25",
             "mean_tv": mean_tv_b, "mean_moran": I_b},
        ]

        # Grid projection
        g_tv_a, lc, ltc = project_to_grid(df_a["lon"].values, df_a["lat"].values, tv_a, res)
        g_tv_b, _,  _   = project_to_grid(df_b["lon"].values, df_b["lat"].values, tv_b, res)
        g_li_a, _,  _   = project_to_grid(df_a["lon"].values, df_a["lat"].values, li_a, res)
        g_li_b, _,  _   = project_to_grid(df_b["lon"].values, df_b["lat"].values, li_b, res)

        # Shared scales: vmax = 98th percentile across both models
        vmax_tv = max(
            np.nanpercentile(tv_a[~np.isnan(tv_a)], 98),
            np.nanpercentile(tv_b[~np.isnan(tv_b)], 98),
        )
        vmax_li = max(
            np.nanpercentile(np.abs(li_a[~np.isnan(li_a)]), 98),
            np.nanpercentile(np.abs(li_b[~np.isnan(li_b)]), 98),
        )

        # TV comparison
        plot_comparison(
            g_tv_a, g_tv_b, lc, ltc,
            args.name_a, args.name_b, view, "Local TV (inv25)",
            cmap="YlOrRd", vmin=0, vmax=vmax_tv, center=False,
            vals_a=tv_a, vals_b=tv_b,
            out_path=out_dir / f"compare_tv_{view}.png",
        )

        # Moran comparison
        plot_comparison(
            g_li_a, g_li_b, lc, ltc,
            args.name_a, args.name_b, view, "Local Moran's I (inv25)",
            cmap="RdBu_r", vmin=-vmax_li, vmax=vmax_li, center=True,
            vals_a=li_a, vals_b=li_b,
            out_path=out_dir / f"compare_moran_{view}.png",
        )

        print(f"  -> compare_tv_{view}.png  compare_moran_{view}.png", flush=True)

    # Summary CSV
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    print(f"\n{df_summary.to_string(index=False)}")
    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main(parse_args())