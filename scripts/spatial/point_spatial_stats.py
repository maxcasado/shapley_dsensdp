"""
compute_point_spatial_stats.py
------------------------------
Point-level spatial statistics on Shapley values.

Unlike compute_spatial_stats.py which projects onto a regular grid,
this script computes statistics directly at each sample point using
geographic neighbor search (BallTree + haversine).

Three spatial weight matrices:
    knn5        : 5 nearest neighbors (binary weights, row-normalised)
    exp25       : exp(-d_km / sigma) for d < 25 km  (sigma=8 km by default)
    inv25       : 1/d_km               for d < 25 km

Statistics computed per point:
    local_tv    : mean |phi_i - phi_j| over neighbours
    local_moran : LISA  I_i = z_i * sum_j(w_ij * z_j)  (Anselin 1995)

Outputs:
    {out_dir}/
        point_stats_{view}_{weight}.csv     local TV + local Moran per point
        summary.csv                         global mean TV + Moran per model/view/weight
        map_tv_{view}_{weight}.png          geographic map of local TV
        map_moran_{view}_{weight}.png       geographic map of local Moran's I

Usage:
    python -m scripts.spatial.point_spatial_stats \\
        --csv preds/spatial/shapley_spatial.csv \\
        --out results/point_stats \\
        --sigma 8.0 \\
        --threshold_km 25.0 \\
        --knn 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="Path to shapley_spatial.csv")
    p.add_argument("--out", required=True,
                   help="Output directory")
    p.add_argument("--threshold_km", type=float, default=25.0,
                   help="Distance threshold for exp/inv weights (km, default 25)")
    p.add_argument("--sigma", type=float, default=8.0,
                   help="Scale for exp(-d/sigma) kernel (km, default 8)")
    p.add_argument("--knn", type=int, default=5,
                   help="Number of nearest neighbours for knn weight (default 5)")
    p.add_argument("--signed", action="store_true",
                   help="Use signed phi (default: |phi|)")
    p.add_argument("--grid_resolution", type=float, default=None,
                   help="If set, also produce gridded maps at this resolution in degrees "
                        "(e.g. 2.0). Uses mean aggregation per cell.")
    p.add_argument("--model_name", default="",
                   help="Model name for summary CSV (optional)")
    return p.parse_args()


# ── Haversine distance ────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    """Scalar haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ── Weight matrices ───────────────────────────────────────────────────────────

def build_weight_matrix(lats, lons, mode, threshold_km=25.0, sigma=8.0, knn=5):
    """
    Build a sparse row-normalised spatial weight matrix.

    Args:
        lats, lons   : arrays of coordinates (degrees)
        mode         : "knn5" | "exp25" | "inv25"
        threshold_km : distance threshold for exp/inv modes
        sigma        : bandwidth for exp kernel (km)
        knn          : number of neighbours for knn mode

    Returns:
        indices : list of arrays — neighbour indices per point
        weights : list of arrays — normalised weights per point
    """
    n = len(lats)
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
    tree = BallTree(coords_rad, metric="haversine")

    indices = []
    weights = []

    if mode == "knn5":
        # k+1 because BallTree returns the point itself as nearest neighbour
        dist_rad, idx = tree.query(coords_rad, k=knn + 1)
        for i in range(n):
            nb_idx = idx[i, 1:]           # exclude self
            w = np.ones(knn, dtype=np.float32)
            w /= w.sum()
            indices.append(nb_idx)
            weights.append(w)

    elif mode in ("exp25", "inv25"):
        threshold_rad = threshold_km / EARTH_RADIUS_KM
        idx_list = tree.query_radius(coords_rad, r=threshold_rad,
                                     return_distance=True, sort_results=True)
        idx_arr, dist_arr = idx_list[1], idx_list[0]

        for i in range(n):
            mask   = idx_arr[i] != i      # exclude self
            nb_idx = idx_arr[i][mask].astype(int)
            d_km   = dist_arr[i][mask] * EARTH_RADIUS_KM

            if len(nb_idx) == 0:
                indices.append(np.array([], dtype=int))
                weights.append(np.array([], dtype=np.float32))
                continue

            if mode == "exp25":
                w = np.exp(-d_km / sigma).astype(np.float32)
            else:  # inv25
                w = (1.0 / np.maximum(d_km, 1e-6)).astype(np.float32)

            w_sum = w.sum()
            if w_sum == 0:
                indices.append(np.array([], dtype=int))
                weights.append(np.array([], dtype=np.float32))
                continue

            w /= w_sum
            indices.append(nb_idx)
            weights.append(w)

    return indices, weights


# ── Local TV ──────────────────────────────────────────────────────────────────

def local_tv(phi, indices, weights):
    """
    Local Total Variation: weighted mean absolute difference to neighbours.

    TV_i = sum_j w_ij |phi_i - phi_j|

    Returns array of shape (n,).  NaN for isolated points (no neighbours).
    """
    n = len(phi)
    tv = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        nb = indices[i]
        w  = weights[i]
        if len(nb) == 0:
            continue
        tv[i] = np.dot(w, np.abs(phi[i] - phi[nb]))
    return tv


# ── Local Moran's I (LISA) ────────────────────────────────────────────────────

def local_moran(phi, indices, weights):
    """
    Local Indicator of Spatial Association (LISA).

    I_i = z_i * sum_j w_ij * z_j

    where z_i = (phi_i - mean) / std.

    Returns:
        lisa    : array (n,) of local Moran values
        global_I: scalar global Moran's I = mean(lisa) * n / sum_W
    """
    mu  = np.nanmean(phi)
    std = np.nanstd(phi)
    if std == 0:
        return np.zeros(len(phi)), 0.0

    z    = (phi - mu) / std
    n    = len(phi)
    lisa = np.full(n, np.nan, dtype=np.float64)
    W_sum = 0.0

    for i in range(n):
        nb = indices[i]
        w  = weights[i]
        if len(nb) == 0:
            continue
        lisa[i] = z[i] * np.dot(w, z[nb])
        W_sum   += w.sum()

    # Global Moran's I = n / W * sum(I_i) / sum(z_i^2)
    global_I = np.nanmean(lisa)   # simplified; exact formula requires W
    return lisa.astype(np.float32), float(global_I)


# ── Maps ──────────────────────────────────────────────────────────────────────

def _project_to_grid(lons, lats, values, resolution):
    """Bin point values onto a regular lon/lat grid, mean per cell."""
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_edges = np.arange(-90,   90 + resolution, resolution)
    n_lon = len(lon_edges) - 1
    n_lat = len(lat_edges) - 1

    lon_idx = np.clip(np.digitize(lons, lon_edges) - 1, 0, n_lon - 1)
    lat_idx = np.clip(np.digitize(lats, lat_edges) - 1, 0, n_lat - 1)
    flat    = lon_idx * n_lat + lat_idx

    valid   = ~np.isnan(values)
    grid_sum   = np.bincount(flat[valid], weights=values[valid],
                              minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    grid_count = np.bincount(flat[valid],
                              minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    grid = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    return grid, lon_centers, lat_centers


def _plot_grid_map(grid, lon_centers, lat_centers, title, label, cmap,
                   out_path, vmin=None, vmax=None, center=False):
    """pcolormesh map of a gridded field."""
    if vmax is None:
        vmax = np.nanpercentile(np.abs(grid), 98)
    if center:
        vmin = -vmax
    elif vmin is None:
        vmin = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(lon_centers, lat_centers, grid.T,
                       cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(pc, ax=ax, label=label, fraction=0.03, pad=0.04, shrink=0.7)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_point_map(lons, lats, values, title, label, cmap, out_path,
                    vmin=None, vmax=None, center=False):
    """Rasterized scatter map of per-point values."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(lons))

    if vmax is None:
        vmax = np.nanpercentile(np.abs(values), 98)
    if center:
        vmin = -vmax
    elif vmin is None:
        vmin = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(lons[idx], lats[idx], c=values[idx],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=0.5, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label=label, fraction=0.03, pad=0.04, shrink=0.7)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

WEIGHT_MODES = ["knn5", "exp25", "inv25"]

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.csv}...", flush=True)
    df = pd.read_csv(args.csv)
    lats = df["lat"].values
    lons = df["lon"].values
    view_names = [c.replace("_shapley", "")
                  for c in df.columns if c.endswith("_shapley")]
    phi_label = "φ" if args.signed else "|φ|"
    n = len(df)
    print(f"  {n} points, {len(view_names)} modalities: {view_names}\n")

    # ── Build weight matrices ─────────────────────────────────────────────────
    W = {}
    for mode in WEIGHT_MODES:
        print(f"Building weight matrix: {mode}...", flush=True)
        W[mode] = build_weight_matrix(
            lats, lons, mode=mode,
            threshold_km=args.threshold_km,
            sigma=args.sigma,
            knn=args.knn,
        )
        nb_counts = [len(W[mode][0][i]) for i in range(n)]
        print(f"  mean neighbours: {np.mean(nb_counts):.1f}  "
              f"max: {np.max(nb_counts)}", flush=True)

    # ── Per view × per weight mode ────────────────────────────────────────────
    summary_rows = []

    for view in view_names:
        col    = f"{view}_shapley"
        phi    = df[col].values if args.signed else df[col].abs().values

        for mode in WEIGHT_MODES:
            nb_idx, nb_w = W[mode]
            print(f"\n── {view} × {mode} ──────────────────────────────")

            # Local TV
            print(f"  Computing local TV...", flush=True)
            tv = local_tv(phi, nb_idx, nb_w)
            mean_tv = float(np.nanmean(tv))
            print(f"  mean TV = {mean_tv:.6f}", flush=True)

            # Local Moran's I
            print(f"  Computing local Moran's I...", flush=True)
            lisa, global_I = local_moran(phi, nb_idx, nb_w)
            print(f"  global Moran's I = {global_I:.4f}", flush=True)

            # Save point-level CSV
            df_out = pd.DataFrame({
                "lon": lons, "lat": lats,
                f"phi_{view}": phi,
                "local_tv":    tv,
                "local_moran": lisa,
            })
            csv_path = out_dir / f"point_stats_{view}_{mode}.csv"
            df_out.to_csv(csv_path, index=False, float_format="%.6f")

            # Summary row
            summary_rows.append({
                "model":    args.model_name,
                "view":     view,
                "weight":   mode,
                "mean_tv":  mean_tv,
                "mean_moran": global_I,
                "n_points": int((~np.isnan(tv)).sum()),
            })

            # Maps
            _plot_point_map(
                lons, lats, tv,
                title=f"Local TV — {view} ({mode})",
                label="TV locale", cmap="YlOrRd",
                out_path=out_dir / f"map_tv_{view}_{mode}.png",
            )
            _plot_point_map(
                lons, lats, lisa,
                title=f"Local Moran's I — {view} ({mode})",
                label="LISA", cmap="RdBu_r",
                out_path=out_dir / f"map_moran_{view}_{mode}.png",
                center=True,
            )

            # Gridded versions if requested
            if args.grid_resolution:
                res = args.grid_resolution
                g_tv, lc, ltc = _project_to_grid(lons, lats, tv, res)
                _plot_grid_map(g_tv, lc, ltc,
                               title=f"Local TV (grille {res}°) — {view} ({mode})",
                               label="TV locale (moy. cellule)", cmap="YlOrRd",
                               out_path=out_dir / f"grid_tv_{view}_{mode}.png")

                g_moran, _, _ = _project_to_grid(lons, lats, lisa, res)
                _plot_grid_map(g_moran, lc, ltc,
                               title=f"Local Moran's I (grille {res}°) — {view} ({mode})",
                               label="LISA (moy. cellule)", cmap="RdBu_r",
                               out_path=out_dir / f"grid_moran_{view}_{mode}.png",
                               center=True)
                print(f"  -> gridded maps saved ({res}°)", flush=True)

            print(f"  -> maps saved", flush=True)

    # ── Global summary ────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")

    sep = "=" * 72
    print(f"\n{sep}\n  Summary\n{sep}")
    print(df_summary.to_string(index=False))
    print(sep)
    print(f"\nAll outputs saved -> {out_dir}")


if __name__ == "__main__":
    main(parse_args())