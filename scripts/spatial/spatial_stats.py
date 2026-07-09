"""
compute_spatial_stats.py
------------------------
Spatial statistics on Shapley values projected onto a geographic grid.

Pipeline:
    1. Load shapley_spatial.csv (output of explain.py)
    2. Project φ_m onto a regular lon/lat grid (mean per cell)
    3. Compute Moran's I(φ_m)  — spatial autocorrelation
    4. Compute TV(φ_m)          — total variation (spatial roughness)
    5. Save results + visualisations

Moran's I is computed on the gridded data using queen contiguity weights
(each cell connected to its 8 neighbours). A Monte Carlo permutation test
provides the p-value.

Total Variation is the L1 norm of the discrete gradient on the grid:
    TV(f) = Σᵢⱼ |f(i,j) - f(i+1,j)| + |f(i,j) - f(i,j+1)|

Usage:
    python -m scripts.spatial.spatial_stats \\
        --csv preds/spatial/shapley_spatial.csv \\
        --resolution 2.0 \\
        --n_permutations 999 \\
        --out preds/spatial/spatial_stats
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default="preds/spatial/shapley_spatial.csv",
                   help="Path to shapley_spatial.csv")
    p.add_argument("--resolution", type=float, default=2.0,
                   help="Grid cell size in degrees (default 2.0°)")
    p.add_argument("--n_permutations", type=int, default=999,
                   help="Monte Carlo permutations for Moran's I p-value")
    p.add_argument("--out", default="preds/spatial/spatial_stats",
                   help="Output directory")
    p.add_argument("--signed", action="store_true",
                   help="Use signed φ (default: |φ|)")
    p.add_argument("--pred_crop", action="store_true",
                   help="Only include points where model predicts crop (pred_score > 0)")
    return p.parse_args()


# ── Grid projection ───────────────────────────────────────────────────────────

def project_to_grid(lons, lats, values, resolution):
    """
    Bin point values onto a regular lon/lat grid, computing the mean per cell.

    Returns:
        grid   : (n_lon, n_lat) array, NaN for empty cells
        lon_centers : 1D array of cell centre longitudes
        lat_centers : 1D array of cell centre latitudes
        n_points    : (n_lon, n_lat) count of points per cell
    """
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_edges = np.arange(-90,   90 + resolution, resolution)
    n_lon     = len(lon_edges) - 1
    n_lat     = len(lat_edges) - 1

    lon_idx = np.clip(np.digitize(lons, lon_edges) - 1, 0, n_lon - 1)
    lat_idx = np.clip(np.digitize(lats, lat_edges) - 1, 0, n_lat - 1)
    flat    = lon_idx * n_lat + lat_idx

    grid_sum   = np.bincount(flat, weights=values,
                              minlength=n_lon * n_lat).reshape(n_lon, n_lat)
    grid_count = np.bincount(flat, weights=np.ones_like(values),
                              minlength=n_lon * n_lat).reshape(n_lon, n_lat)

    grid = np.where(grid_count > 0, grid_sum / grid_count, np.nan)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    return grid, lon_centers, lat_centers, grid_count


# ── Moran's I ─────────────────────────────────────────────────────────────────

def morans_i(grid, n_permutations=999, seed=0):
    """
    Compute Moran's I on a 2D grid using queen contiguity weights.

    Only non-NaN cells are included. Empty cells (NaN) are excluded from
    both the weight matrix and the statistic.

    Args:
        grid:           (n_lon, n_lat) array with NaN for missing cells.
        n_permutations: Monte Carlo permutations for p-value (0 = skip).
        seed:           RNG seed for permutations.

    Returns:
        dict with keys: I, E_I, Var_I, z_score, p_value, n_cells
    """
    n_lon, n_lat = grid.shape

    # Flatten and keep only valid cells
    flat     = grid.flatten()
    valid    = ~np.isnan(flat)
    idx_flat = np.where(valid)[0]
    n        = len(idx_flat)

    if n < 4:
        return {"I": np.nan, "E_I": np.nan, "Var_I": np.nan,
                "z_score": np.nan, "p_value": np.nan, "n_cells": n}

    x     = flat[idx_flat]
    x_bar = x.mean()
    x_dev = x - x_bar

    # Map flat index → (i, j) in grid
    rows = idx_flat // n_lat
    cols = idx_flat  % n_lat

    # Build sparse queen-contiguity weight matrix
    # For each valid cell, find its valid neighbours
    # We store as lists for efficiency
    cell_to_pos = {idx: pos for pos, idx in enumerate(idx_flat)}

    W_sum  = 0.0
    num    = 0.0

    for pos_i, (r, c) in enumerate(zip(rows, cols)):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_lon and 0 <= nc < n_lat:
                    nb_flat = nr * n_lat + nc
                    if nb_flat in cell_to_pos:
                        pos_j  = cell_to_pos[nb_flat]
                        num   += x_dev[pos_i] * x_dev[pos_j]
                        W_sum += 1.0

    denom = np.sum(x_dev ** 2)
    if denom == 0 or W_sum == 0:
        return {"I": np.nan, "E_I": np.nan, "Var_I": np.nan,
                "z_score": np.nan, "p_value": np.nan, "n_cells": n}

    I     = (n / W_sum) * (num / denom)
    E_I   = -1.0 / (n - 1)

    # Monte Carlo p-value
    p_value = np.nan
    if n_permutations > 0:
        rng    = np.random.default_rng(seed)
        sim_I  = np.empty(n_permutations)
        for k in range(n_permutations):
            xp    = rng.permutation(x)
            xp_d  = xp - xp.mean()
            num_p = 0.0
            for pos_i, (r, c) in enumerate(zip(rows, cols)):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_lon and 0 <= nc < n_lat:
                            nb_flat = nr * n_lat + nc
                            if nb_flat in cell_to_pos:
                                pos_j  = cell_to_pos[nb_flat]
                                num_p += xp_d[pos_i] * xp_d[pos_j]
            denom_p = np.sum(xp_d ** 2)
            sim_I[k] = (n / W_sum) * (num_p / denom_p) if denom_p > 0 else 0
        p_value = (np.sum(np.abs(sim_I) >= np.abs(I)) + 1) / (n_permutations + 1)

    return {
        "I":        I,
        "E_I":      E_I,
        "Var_I":    np.var(sim_I) if n_permutations > 0 else np.nan,
        "z_score":  (I - E_I) / np.std(sim_I) if n_permutations > 0 else np.nan,
        "p_value":  p_value,
        "n_cells":  n,
    }


# ── Total Variation ───────────────────────────────────────────────────────────

def total_variation(grid):
    """
    Isotropic TV on a 2D grid, ignoring NaN boundaries.

    TV(f) = Σᵢⱼ √( (f(i+1,j)-f(i,j))² + (f(i,j+1)-f(i,j))² )

    Returns scalar TV and normalised TV (divided by number of valid pairs).
    """
    g = np.where(np.isnan(grid), 0.0, grid)   # treat NaN as 0 for differences

    dh = np.diff(g, axis=0)   # horizontal gradient (along lon)
    dv = np.diff(g, axis=1)   # vertical gradient   (along lat)

    # Mask pairs where either cell is NaN
    mask_h = ~(np.isnan(grid[:-1, :]) | np.isnan(grid[1:, :]))
    mask_v = ~(np.isnan(grid[:, :-1]) | np.isnan(grid[:, 1:]))

    tv_h = np.abs(dh[mask_h]).sum()
    tv_v = np.abs(dv[mask_v]).sum()
    tv   = tv_h + tv_v

    n_pairs = mask_h.sum() + mask_v.sum()
    tv_norm = tv / n_pairs if n_pairs > 0 else np.nan

    return float(tv), float(tv_norm)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _plot_grid(grid, lon_centers, lat_centers, title, cmap, out_path,
               vmin=None, vmax=None, label="φ"):
    """Plot a gridded Shapley map."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(lon_centers, lat_centers, grid.T,
                       cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax)
    plt.colorbar(pc, ax=ax, label=label, fraction=0.03, pad=0.04, shrink=0.7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tv_map(grid, lon_centers, lat_centers, view, out_path):
    """
    Plot the local TV contribution G(i,j) = sqrt(dh^2 + dv^2) for each cell.
    Shows where spatial roughness is concentrated geographically.
    """
    n_lon, n_lat = grid.shape
    G = np.full((n_lon, n_lat), np.nan)

    for i in range(n_lon - 1):
        for j in range(n_lat - 1):
            if not (np.isnan(grid[i, j]) or
                    np.isnan(grid[i+1, j]) or
                    np.isnan(grid[i, j+1])):
                dh = grid[i+1, j] - grid[i, j]
                dv = grid[i, j+1] - grid[i, j]
                G[i, j] = np.sqrt(dh**2 + dv**2)

    vmax = np.nanpercentile(G, 98)
    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(lon_centers, lat_centers, G.T,
                       cmap="YlOrRd", shading="auto",
                       vmin=0, vmax=vmax)
    plt.colorbar(pc, ax=ax, label="TV locale G(i,j)", fraction=0.03,
                 pad=0.04, shrink=0.7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Rugosité spatiale locale — {view}", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    """
    Barplot comparing Moran's I and normalised TV across modalities.
    """
    out_dir = Path(out_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x       = np.arange(len(view_names))
    colors  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    moran_I = [results[v]["moran_I"]   for v in view_names]
    tv_norm = [results[v]["tv_norm"]   for v in view_names]
    pvals   = [results[v]["p_value"]   for v in view_names]

    bars1 = ax1.bar(x, moran_I, color=colors, edgecolor="white", zorder=2)
    ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x); ax1.set_xticklabels(view_names, fontsize=10)
    ax1.set_ylabel("Moran's I", fontsize=11)
    ax1.set_title("Spatial autocorrelation", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # Significance stars
    for xi, (mi, pv) in enumerate(zip(moran_I, pvals)):
        if pv < 0.001:   stars = "***"
        elif pv < 0.01:  stars = "**"
        elif pv < 0.05:  stars = "*"
        else:             stars = "ns"
        offset = max(moran_I) * 0.03
        ax1.text(xi, mi + offset, stars, ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    bars2 = ax2.bar(x, tv_norm, color=colors, edgecolor="white", zorder=2)
    ax2.set_xticks(x); ax2.set_xticklabels(view_names, fontsize=10)
    ax2.set_ylabel("TV normalisée", fontsize=11)
    ax2.set_title("Variation totale (rugosité spatiale)", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_dir / "spatial_stats_summary.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"  -> spatial_stats_summary.png", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.csv}...", flush=True)
    df = pd.read_csv(args.csv)

    phi_label  = "φ" if args.signed else "|φ|"
    view_names = [c.replace("_shapley", "")
                  for c in df.columns if c.endswith("_shapley")]
    print(f"  {len(df)} points, {len(view_names)} modalities: {view_names}")

    if args.pred_crop:
        if "pred_score" not in df.columns:
            print("ERROR: --pred_crop requires 'pred_score' column in CSV")
            return
        n_before = len(df)
        df = df[df["pred_score"] > 0].reset_index(drop=True)
        print(f"  Filtered to pred_crop: {len(df)} / {n_before} points "
              f"({len(df)/n_before*100:.1f}%)")

    lons = df["lon"].values
    lats = df["lat"].values

    print(f"  Grid resolution: {args.resolution}°  "
          f"  Mode: {phi_label}  "
          f"  Permutations: {args.n_permutations}\n")

    results = {}

    # ── Per-modality ──────────────────────────────────────────────────────────
    for view in view_names:
        col    = f"{view}_shapley"
        values = df[col].values if args.signed else df[col].abs().values

        print(f"── {view} ────────────────────────────────────")

        # Grid projection
        grid, lon_c, lat_c, counts = project_to_grid(
            lons, lats, values, args.resolution)
        n_cells = int((~np.isnan(grid)).sum())
        print(f"  Grid cells populated: {n_cells} / "
              f"{grid.shape[0]*grid.shape[1]}", flush=True)

        # Moran's I
        print(f"  Computing Moran's I ({args.n_permutations} permutations)...",
              flush=True)
        moran = morans_i(grid, n_permutations=args.n_permutations)
        print(f"  I={moran['I']:.4f}  E[I]={moran['E_I']:.4f}  "
              f"p={moran['p_value']:.4f}", flush=True)

        # Total Variation
        tv, tv_norm = total_variation(grid)
        print(f"  TV={tv:.4f}  TV_norm={tv_norm:.6f}", flush=True)

        results[view] = {
            "moran_I":  moran["I"],
            "E_I":      moran["E_I"],
            "z_score":  moran["z_score"],
            "p_value":  moran["p_value"],
            "n_cells":  moran["n_cells"],
            "tv":       tv,
            "tv_norm":  tv_norm,
        }

        # Grid map
        vmax = np.nanpercentile(np.abs(grid), 98)
        cmap = "RdBu_r" if args.signed else "plasma"
        vmin = -vmax if args.signed else 0
        _plot_grid(grid, lon_c, lat_c,
                   title=f"{phi_label}  {view}  "
                         f"(Moran's I={moran['I']:.3f}, p={moran['p_value']:.3f})",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   label=phi_label,
                   out_path=out_dir / f"grid_{view}.png")

        plot_tv_map(grid, lon_c, lat_c, view,
                    out_path=out_dir / f"tv_map_{view}.png")
        print(f"  -> tv_map_{view}.png\n", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}\n  Spatial statistics summary\n{sep}")
    print(f"  {'Modality':<14}  {'Moran I':>8}  {'z-score':>8}  "
          f"{'p-value':>8}  {'TV_norm':>10}")
    print(f"  {'-'*60}")
    for v in view_names:
        r = results[v]
        print(f"  {v:<14}  {r['moran_I']:>8.4f}  {r['z_score']:>8.2f}  "
              f"{r['p_value']:>8.4f}  {r['tv_norm']:>10.6f}")
    print(sep)

    # CSV
    df_out = pd.DataFrame([
        {"view": v, **results[v]} for v in view_names
    ])
    df_out.to_csv(out_dir / "spatial_stats.csv", index=False, float_format="%.6f")
    print(f"\nCSV saved -> {out_dir}/spatial_stats.csv")

    # Summary plot
    plot_summary(results, view_names, out_dir)


if __name__ == "__main__":
    main(parse_args())