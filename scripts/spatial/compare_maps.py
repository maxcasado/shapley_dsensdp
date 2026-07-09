"""
compare_shapley_maps.py
-----------------------
Side-by-side Shapley maps for two models with shared colorbars per modality.

Usage:
    python -m scripts.spatial.compare_maps \
        --csv_a preds_com/spatial/shapley_spatial.csv \
        --csv_b preds_com_geo_fixed/spatial/shapley_spatial.csv \
        --name_a com --name_b com_geo \
        --out results/compare_shapley_maps \
        --resolution 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a",       required=True)
    p.add_argument("--csv_b",       required=True)
    p.add_argument("--name_a",      required=True)
    p.add_argument("--name_b",      required=True)
    p.add_argument("--out",         required=True)
    p.add_argument("--resolution",  type=float, default=2.0)
    p.add_argument("--signed",      action="store_true")
    return p.parse_args()


def to_grid_mean_std(lons, lats, values, resolution):
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_edges = np.arange(-90,   90 + resolution, resolution)
    n_lon = len(lon_edges) - 1
    n_lat = len(lat_edges) - 1
    lon_idx = np.clip(np.digitize(lons, lon_edges) - 1, 0, n_lon - 1)
    lat_idx = np.clip(np.digitize(lats, lat_edges) - 1, 0, n_lat - 1)
    flat    = lon_idx * n_lat + lat_idx
    valid   = ~np.isnan(values)
    gs  = np.bincount(flat[valid], weights=values[valid],
                      minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    gs2 = np.bincount(flat[valid], weights=values[valid]**2,
                      minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    gc  = np.bincount(flat[valid], minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    mean = np.where(gc > 0, gs / gc, np.nan)
    std  = np.where(gc >= 2,
                    np.sqrt(np.maximum(gs2 / gc - (gs / gc)**2, 0)),
                    np.nan)
    lc  = (lon_edges[:-1] + lon_edges[1:]) / 2
    ltc = (lat_edges[:-1] + lat_edges[1:]) / 2
    return mean, std, lc, ltc


def plot_comparison_map(grid_a, grid_b, lc, ltc,
                        name_a, name_b, view, stat,
                        cmap, vmin, vmax, out_path):
    """Two pcolormesh maps side by side with shared colorbar."""
    fig = plt.figure(figsize=(16, 4.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[8, 8, 0.4], wspace=0.06)
    ax_a    = fig.add_subplot(gs[0])
    ax_b    = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    kw = dict(cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

    for ax, grid, name in [(ax_a, grid_a, name_a), (ax_b, grid_b, name_b)]:
        pc = ax.pcolormesh(lc, ltc, grid.T, **kw)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name} — {view} ({stat})", fontsize=11)
        ax.set_xlabel("Longitude", fontsize=9)
    ax_a.set_ylabel("Latitude", fontsize=9)
    ax_b.set_yticks([])

    cb = fig.colorbar(pc, cax=ax_cbar)
    cb.set_label(stat, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)

    # Common views
    views_a = [c.replace("_shapley", "") for c in df_a.columns if c.endswith("_shapley")]
    views_b = [c.replace("_shapley", "") for c in df_b.columns if c.endswith("_shapley")]
    view_names = [v for v in views_a if v in views_b]
    print(f"Common views: {view_names}")

    phi_label = "φ" if args.signed else "|φ|"

    for view in view_names:
        col = f"{view}_shapley"
        vals_a = df_a[col].values if args.signed else df_a[col].abs().values
        vals_b = df_b[col].values if args.signed else df_b[col].abs().values

        mean_a, std_a, lc, ltc = to_grid_mean_std(
            df_a["lon"].values, df_a["lat"].values, vals_a, res)
        mean_b, std_b, _,  _   = to_grid_mean_std(
            df_b["lon"].values, df_b["lat"].values, vals_b, res)

        # Shared vmin/vmax across both models
        all_mean = np.concatenate([mean_a[~np.isnan(mean_a)],
                                   mean_b[~np.isnan(mean_b)]])
        vmax_mean = np.percentile(all_mean, 98)
        vmin_mean = 0 if not args.signed else -vmax_mean

        all_std = np.concatenate([std_a[~np.isnan(std_a)],
                                  std_b[~np.isnan(std_b)]])
        vmax_std = np.percentile(all_std, 98) if len(all_std) > 0 else 1.0

        print(f"  {view}: mean vmax={vmax_mean:.4f}  std vmax={vmax_std:.4f}")

        # Mean comparison
        plot_comparison_map(
            mean_a, mean_b, lc, ltc,
            args.name_a, args.name_b, view,
            f"mean {phi_label}",
            cmap="plasma" if not args.signed else "RdBu_r",
            vmin=vmin_mean, vmax=vmax_mean,
            out_path=out_dir / f"compare_mean_{view}.png",
        )

        # Std comparison
        plot_comparison_map(
            std_a, std_b, lc, ltc,
            args.name_a, args.name_b, view,
            f"std {phi_label}",
            cmap="YlOrRd",
            vmin=0, vmax=vmax_std,
            out_path=out_dir / f"compare_std_{view}.png",
        )

        print(f"  -> compare_mean_{view}.png  compare_std_{view}.png")

    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main(parse_args())