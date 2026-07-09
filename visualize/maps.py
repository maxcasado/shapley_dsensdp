"""
visualize/maps.py
-----------------
Geographic visualization functions for Shapley maps, GT/prediction maps,
interaction graphs and matrices, and grouped statistics.

All functions read from a pandas DataFrame built by infer_france.py and
write PNG files to the specified output directories.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Cartographic helpers ──────────────────────────────────────────────────────

def _get_world_borders():
    """
    Load country borders for map background.
    Tries cartopy, then geopandas/naturalearth, returns None if unavailable.
    """
    try:
        import cartopy.feature as cfeature
        return ("cartopy", cfeature.BORDERS, cfeature.COASTLINE)
    except ImportError:
        pass

    try:
        import geopandas as gpd
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        return ("geopandas_land", world)
    except Exception:
        pass

    try:
        import geopandas as gpd
        from pathlib import Path as _Path
        _NE_SHP = _Path.home() / ".cache" / "naturalearth" / "ne_110m_admin_0_countries.shp"
        if not _NE_SHP.exists():
            raise FileNotFoundError(_NE_SHP)
        return ("geopandas", gpd.read_file(_NE_SHP))
    except Exception:
        pass

    return None


def _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max):
    if borders is None:
        return
    kind = borders[0]
    if kind == "cartopy":
        ax.add_feature(borders[1], linewidth=0.6, edgecolor="black")
        ax.add_feature(borders[2], linewidth=0.5, edgecolor="black")
    elif kind in ("geopandas", "geopandas_land"):
        borders[1].boundary.plot(ax=ax, color="black", linewidth=0.5)


def _setup_ax(ax, borders, title, lon_min, lon_max, lat_min, lat_max):
    _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max)
    ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
    ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=13)


def _scatter_discrete(ax, lons, lats, values, groups, alpha=0.8):
    """
    Scatter plot with discrete categories, randomising point order to avoid
    one category masking the others.

    groups: list of (value, label, color)
    Returns legend handles.
    """
    rng    = np.random.default_rng(0)
    idx    = rng.permutation(len(lons))
    color_map = {val: color for val, _, color in groups}
    colors    = np.array([color_map[v] for v in values[idx]])
    ax.scatter(lons[idx], lats[idx], c=colors, marker=',', linewidths=0, alpha=alpha)
    return [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=7, label=name)
        for _, name, color in groups
    ]


# ── Palettes ──────────────────────────────────────────────────────────────────

CONTINENT_COLORS = {
    "Europe":           "#4e79a7",
    "Asia":             "#f28e2b",  "Asie":             "#f28e2b",
    "Africa":           "#e15759",  "Afrique":          "#e15759",
    "North America":    "#76b7b2",  "Amerique du Nord": "#76b7b2",
    "South America":    "#59a14f",  "Amerique du Sud":  "#59a14f",
    "Oceania":          "#edc948",  "Oceanie":          "#edc948",
    "Antarctica":       "#b07aa1",  "Antarctique":      "#b07aa1",
    "Unknown":          "#bab0ac",  "Inconnu":          "#bab0ac",
}


# ── Shapley maps ──────────────────────────────────────────────────────────────

def plot_shapley_maps(df: pd.DataFrame, view_names: list, out_dir,
                      lon_min, lon_max, lat_min, lat_max,
                      resolution: float = 2.0):
    """
    Shapley maps using pcolormesh on a regular grid (mean per cell).
    resolution: grid cell size in degrees (default 2.0°).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    lons = df["lon"].values
    lats = df["lat"].values

    def _to_grid(values, res):
        lon_edges = np.arange(-180, 180 + res, res)
        lat_edges = np.arange(-90,   90 + res, res)
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
        # std = sqrt(E[x²] - E[x]²), only for cells with >=2 points
        std  = np.where(gc >= 2,
                        np.sqrt(np.maximum(gs2 / gc - (gs / gc)**2, 0)),
                        np.nan)
        lc  = (lon_edges[:-1] + lon_edges[1:]) / 2
        ltc = (lat_edges[:-1] + lat_edges[1:]) / 2
        return mean, std, lc, ltc

    vmax = max(np.nanpercentile(df[f"{v}_shapley"].abs().values, 98)
               for v in view_names)

    # ── Combined mean figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.35})
    axes = axes.flatten()
    for ax, view in zip(axes, view_names):
        values = df[f"{view}_shapley"].abs().values
        mean, std, lc, ltc = _to_grid(values, resolution)
        _setup_ax(ax, borders, view, lon_min, lon_max, lat_min, lat_max)
        pc = ax.pcolormesh(lc, ltc, mean.T, cmap="plasma",
                           vmin=0, vmax=vmax, shading="auto")
        plt.colorbar(pc, ax=ax, fraction=0.03, shrink=0.6, pad=0.04,
                     label="|Shapley value| mean")
    fig.suptitle(f"Shapley values per modality — mean ({resolution}°×{resolution}° grid)",
                 fontsize=15, y=0.98)
    plt.tight_layout()
    p = out_dir / "shapley_map.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined mean map saved -> {p}", flush=True)

    # ── Combined std figure ───────────────────────────────────────────────────
    std_grids = {}
    for view in view_names:
        values = df[f"{view}_shapley"].abs().values
        _, std, lc, ltc = _to_grid(values, resolution)
        std_grids[view] = (std, lc, ltc)

    vmax_std = max(np.nanpercentile(std_grids[v][0], 98)
                   for v in view_names
                   if not np.all(np.isnan(std_grids[v][0])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.35})
    axes = axes.flatten()
    for ax, view in zip(axes, view_names):
        std, lc, ltc = std_grids[view]
        _setup_ax(ax, borders, view, lon_min, lon_max, lat_min, lat_max)
        pc = ax.pcolormesh(lc, ltc, std.T, cmap="YlOrRd",
                           vmin=0, vmax=vmax_std, shading="auto")
        plt.colorbar(pc, ax=ax, fraction=0.03, shrink=0.6, pad=0.04,
                     label="|Shapley value| std")
    fig.suptitle(f"Shapley values per modality — std ({resolution}°×{resolution}° grid)",
                 fontsize=15, y=0.98)
    plt.tight_layout()
    p = out_dir / "shapley_map_std.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined std map saved -> {p}", flush=True)

    # ── Individual figures (mean + std) ──────────────────────────────────────
    for view in view_names:
        values = df[f"{view}_shapley"].abs().values
        mean, std, lc, ltc = _to_grid(values, resolution)
        vmax_v    = np.nanpercentile(values, 98)
        vmax_std_v = np.nanpercentile(std[~np.isnan(std)], 98) \
                     if not np.all(np.isnan(std)) else 1.0

        fig, (ax_m, ax_s) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"wspace": 0.3})
        _setup_ax(ax_m, borders, f"Mean |φ| — {view}",
                  lon_min, lon_max, lat_min, lat_max)
        pc_m = ax_m.pcolormesh(lc, ltc, mean.T, cmap="plasma",
                               vmin=0, vmax=vmax_v, shading="auto")
        plt.colorbar(pc_m, ax=ax_m, fraction=0.03, shrink=0.6, pad=0.04,
                     label="mean |φ|")

        _setup_ax(ax_s, borders, f"Std |φ| — {view}",
                  lon_min, lon_max, lat_min, lat_max)
        pc_s = ax_s.pcolormesh(lc, ltc, std.T, cmap="YlOrRd",
                               vmin=0, vmax=vmax_std_v, shading="auto")
        plt.colorbar(pc_s, ax=ax_s, fraction=0.03, shrink=0.6, pad=0.04,
                     label="std |φ|")

        plt.tight_layout()
        p = out_dir / f"shapley_map_{view}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> {p}", flush=True)


# ── GT / prediction maps ──────────────────────────────────────────────────────

def plot_gt_pred_maps(df: pd.DataFrame, out_dir,
                      lon_min, lon_max, lat_min, lat_max):
    """
    Maps of ground truth labels, predicted score, and classification outcomes
    (correct/incorrect, TP/TN/FP/FN).

    Requires 'label' and 'pred_score' columns in df.
    """
    if "label" not in df.columns or "pred_score" not in df.columns:
        print("WARNING: 'label' or 'pred_score' column missing, map skipped.", flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    GT_COLORS  = [(0, "Non-crop", "#d9534f"), (1, "Crop", "#5cb85c")]
    rng        = np.random.default_rng(0)
    idx        = rng.permutation(len(df))

    # Combined GT + pred score
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"wspace": 0.35})
    _setup_ax(ax_gt,   borders, "Ground truth (crop / non-crop)", lon_min, lon_max, lat_min, lat_max)
    _setup_ax(ax_pred, borders, "Predicted score (all views)",    lon_min, lon_max, lat_min, lat_max)

    handles = _scatter_discrete(ax_gt, df["lon"].values, df["lat"].values,
                                 df["label"].values.astype(int), GT_COLORS)
    ax_gt.legend(handles=handles, loc="lower left", framealpha=0.8)

    sc = ax_pred.scatter(df["lon"].values[idx], df["lat"].values[idx],
                         c=df["pred_score"].values[idx],
                         cmap="RdYlGn", vmin=-1.0, vmax=1.0, marker=",", linewidths=0, alpha=0.8)
    plt.colorbar(sc, ax=ax_pred, fraction=0.03, pad=0.04, label="log-odds P(crop)")

    fig.suptitle("Ground truth vs predicted score", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "gt_pred_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Separate GT
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, borders, "Ground truth (crop / non-crop)", lon_min, lon_max, lat_min, lat_max)
    handles = _scatter_discrete(ax, df["lon"].values, df["lat"].values,
                                 df["label"].values.astype(int), GT_COLORS)
    ax.legend(handles=handles, loc="lower left", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "gt_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Separate pred score
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, borders, "Predicted score (all views)", lon_min, lon_max, lat_min, lat_max)
    sc = ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
                    c=df["pred_score"].values[idx],
                    cmap="RdYlGn", vmin=-1.0, vmax=1.0, marker=",", linewidths=0, alpha=0.8)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="log-odds P(crop)")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Correct / incorrect
    pred_class = (df["pred_score"].values >= 0.5).astype(int)
    label_arr  = df["label"].values.astype(int)
    correct    = pred_class == label_arr
    outcome    = np.where(
        correct & (label_arr == 1), 0,
        np.where(correct & (label_arr == 0), 1,
        np.where(~correct & (label_arr == 0), 2, 3)))

    BIN_COLORS = [(0, "Incorrect", "#d62728"), (1, "Correct", "#2ca02c")]
    ERR_COLORS = [(0, "TP (crop correct)",    "#2ca02c"),
                  (1, "TN (non-crop correct)","#aec7e8"),
                  (2, "FP (false alarm)",     "#d62728"),
                  (3, "FN (missed)",          "#ff7f0e")]

    fig, (ax_bin, ax_err) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"wspace": 0.35})
    _setup_ax(ax_bin, borders, f"Correct vs incorrect (acc={correct.mean():.3f})",
              lon_min, lon_max, lat_min, lat_max)
    _setup_ax(ax_err, borders, "Detail: TP / TN / FP / FN",
              lon_min, lon_max, lat_min, lat_max)

    h_bin = _scatter_discrete(ax_bin, df["lon"].values, df["lat"].values,
                               correct.astype(int), BIN_COLORS)
    h_err = _scatter_discrete(ax_err, df["lon"].values, df["lat"].values,
                               outcome, ERR_COLORS)
    ax_bin.legend(handles=h_bin, loc="lower left", framealpha=0.8)
    ax_err.legend(handles=h_err, loc="lower left", framealpha=0.8)

    fig.suptitle("Classification performance", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "correct_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    for tag, title, groups, arr in [
        ("correct_bin",    f"Correct vs incorrect (acc={correct.mean():.3f})", BIN_COLORS, correct.astype(int)),
        ("correct_detail", "Detail: TP / TN / FP / FN",                        ERR_COLORS, outcome),
    ]:
        fig, ax = plt.subplots(figsize=(6, 7))
        _setup_ax(ax, borders, title, lon_min, lon_max, lat_min, lat_max)
        handles = _scatter_discrete(ax, df["lon"].values, df["lat"].values, arr, groups)
        ax.legend(handles=handles, loc="lower left", framealpha=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_map.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"GT/pred maps saved in {out_dir}", flush=True)


# ── Continent map ─────────────────────────────────────────────────────────────

def plot_continent_map(df: pd.DataFrame, out_dir,
                       lon_min, lon_max, lat_min, lat_max):
    """Map of points coloured by continent."""
    if "continent" not in df.columns:
        print("WARNING: 'continent' column missing, continent map skipped.", flush=True)
        return

    out_dir = Path(out_dir) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    continents_present = sorted(df["continent"].unique())
    groups = [(c, CONTINENT_COLORS.get(c, "#cccccc")) for c in continents_present]

    rng    = np.random.default_rng(0)
    idx    = rng.permutation(len(df))
    color_map = {c: col for c, col in groups}
    colors    = np.array([color_map[c] for c in df["continent"].values[idx]])

    fig, ax = plt.subplots(figsize=(10, 6))
    _setup_ax(ax, borders, f"Points by continent (n={len(df)})",
              lon_min, lon_max, lat_min, lat_max)
    ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
               c=colors, marker=",", linewidths=0, alpha=0.8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=col, markersize=7,
                   label=f"{c} (n={int((df['continent'] == c).sum())})")
        for c, col in groups
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.85,
              fontsize=9, title="Continent", title_fontsize=10)
    plt.tight_layout()
    p = out_dir / "map_continents.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Continent map saved -> {p}", flush=True)


# ── Grouped statistics ────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, view_names: list, out_dir):
    """
    Compute mean(|φ_i|) grouped by: (1) global, (2) continent, (3) correct/incorrect.
    Saves one CSV per grouping to out_dir/stats/.
    """
    stats_dir = Path(out_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    phi_cols = [f"{v}_shapley" for v in view_names]
    sep      = "-" * 60

    def _group_stats(df_sub, group_col):
        rows = []
        for group_val, gdf in df_sub.groupby(group_col):
            row = {group_col: group_val, "n_points": len(gdf)}
            for col, view in zip(phi_cols, view_names):
                vals = gdf[col].abs()
                row[f"|phi|_mean_{view}"] = vals.mean()
                row[f"|phi|_std_{view}"]  = vals.std()
            rows.append(row)
        return pd.DataFrame(rows)

    # 1. Global
    print(f"\n{sep}\n  mean(|phi|) GLOBAL  ({len(df)} points)\n{sep}")
    global_rows = []
    for col, view in zip(phi_cols, view_names):
        vals = df[col].abs()
        print(f"    {view:<12}  mean={vals.mean():.4f}  std={vals.std():.4f}")
        global_rows.append({"modalite": view, "|phi|_mean": vals.mean(),
                             "|phi|_std": vals.std(), "n_points": len(df)})
    pd.DataFrame(global_rows).to_csv(stats_dir / "stats_global.csv", index=False)

    # 2. By continent
    if "continent" in df.columns:
        print(f"\n{sep}\n  mean(|phi|) BY CONTINENT\n{sep}")
        df_cont = _group_stats(df, "continent")
        for _, row in df_cont.iterrows():
            vals_str = "  ".join(f"{v}={row[f'|phi|_mean_{v}']:.4f}" for v in view_names)
            print(f"    {row['continent']:<22} (n={int(row['n_points']):>5})  {vals_str}")
        df_cont.to_csv(stats_dir / "stats_par_continent.csv", index=False)
    else:
        print("  WARNING: 'continent' column missing, per-continent stats skipped.")

    # 3. By correct/incorrect
    if "correct" in df.columns:
        print(f"\n{sep}\n  mean(|phi|) BY PREDICTION (correct / incorrect)\n{sep}")
        df_c = df.copy()
        df_c["prediction"] = df_c["correct"].map({True: "Correct", False: "Incorrect"})
        df_pred = _group_stats(df_c, "prediction")
        for _, row in df_pred.iterrows():
            vals_str = "  ".join(f"{v}={row[f'|phi|_mean_{v}']:.4f}" for v in view_names)
            print(f"    {row['prediction']:<12} (n={int(row['n_points']):>5})  {vals_str}")
        df_pred.to_csv(stats_dir / "stats_par_prediction.csv", index=False)

        if "continent" in df.columns:
            df_c["group"] = df_c["continent"] + " / " + df_c["prediction"]
            _group_stats(df_c, "group").to_csv(
                stats_dir / "stats_continent_x_prediction.csv", index=False)
    else:
        print("  WARNING: 'correct' column missing, per-prediction stats skipped.")

    print(sep)


# ── Interaction graph ─────────────────────────────────────────────────────────

def plot_interaction_graph(shapley_matrix: np.ndarray,
                           interaction_matrix: np.ndarray,
                           view_names: list, out_dir):
    """
    Circular interaction graph: nodes = modalities, edges = mean SII(i,j).
    Blue = synergy (positive SII), red = redundancy (negative SII).
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_views  = len(view_names)
    mean_phi = np.abs(shapley_matrix).mean(axis=0)
    mean_sii = interaction_matrix.mean(axis=0)
    abs_sii  = np.abs(mean_sii)

    angles = [2 * math.pi * k / n_views for k in range(n_views)]
    pos    = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angles)}

    node_sizes = 3000 * mean_phi / (mean_phi.max() + 1e-12)
    edge_max   = abs_sii.max() + 1e-12

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(n_views):
        for j in range(i + 1, n_views):
            val   = mean_sii[i, j]
            lw    = 12 * abs(val) / edge_max
            color = "#2ca02c" if val >= 0 else "#d62728"
            xi, yi = pos[i]; xj, yj = pos[j]
            ax.plot([xi, xj], [yi, yj], color=color, lw=lw, alpha=0.75, zorder=1)
            ax.text((xi+xj)/2, (yi+yj)/2, f"{val:+.3f}",
                    ha="center", va="center", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    for i, name in enumerate(view_names):
        x, y = pos[i]
        ax.scatter(x, y, s=node_sizes[i], color="#4e79a7", zorder=2,
                   edgecolors="white", linewidths=1.5)
        ax.text(x*1.18, y*1.18, name, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x*1.40, y*1.40, f"|φ|={mean_phi[i]:.3f}",
                ha="center", va="center", fontsize=9, color="#666666")

    ax.set_title("Shapley Interaction Graph (SII order 2)", fontsize=13)
    plt.tight_layout()
    p = out_dir / "interaction_graph.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interaction graph saved -> {p}", flush=True)


def plot_interaction_matrix(shapley_matrix: np.ndarray,
                            interaction_matrix: np.ndarray,
                            view_names: list, out_dir):
    """Heatmap of mean SII values (diagonal = mean Shapley values)."""
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n        = len(view_names)
    mean_sii = interaction_matrix.mean(axis=0).copy()
    std_sii  = interaction_matrix.std(axis=0)
    np.fill_diagonal(mean_sii, shapley_matrix.mean(axis=0))
    vmax = np.abs(mean_sii).max()

    fig, ax = plt.subplots(figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))
    im = ax.imshow(mean_sii, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="mean(SII)  [diag = mean(φ)]",
                 fraction=0.04, pad=0.04)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_yticklabels(view_names, fontsize=11)
    ax.set_title("Shapley Interaction Matrix (SII order 2)", fontsize=13)
    for i in range(n):
        for j in range(n):
            val = mean_sii[i, j]
            tc  = "white" if abs(val) > vmax * 0.55 else "#222222"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color=tc, zorder=2)
            if i != j:
                ax.text(j, i + 0.28, f"±{std_sii[i,j]:.3f}",
                        ha="center", va="center", fontsize=7, color=tc, alpha=0.7)
    plt.tight_layout()
    p = out_dir / "interaction_matrix.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interaction matrix saved -> {p}", flush=True)


def plot_interaction_maps(df: pd.DataFrame,
                          interaction_matrix: np.ndarray,
                          view_names: list, out_dir,
                          lon_min, lon_max, lat_min, lat_max):
    """Spatial map of SII for each pair of modalities."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()
    n_views = len(view_names)
    pairs   = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
    n_pairs = len(pairs)
    vmax    = max(np.abs(interaction_matrix[:, i, j]).max() for i, j in pairs)
    rng     = np.random.default_rng(0)
    idx     = rng.permutation(len(df))

    # Combined figure
    ncols = min(3, n_pairs)
    nrows = math.ceil(n_pairs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.35})
    axes = np.array(axes).flatten()
    for ax_idx, (i, j) in enumerate(pairs):
        vi, vj = view_names[i], view_names[j]
        vals   = interaction_matrix[:, i, j]
        _setup_ax(axes[ax_idx], borders, f"SII  {vi} × {vj}",
                  lon_min, lon_max, lat_min, lat_max)
        sc = axes[ax_idx].scatter(df["lon"].values[idx], df["lat"].values[idx],
                                  c=vals[idx], cmap="RdBu",
                                  vmin=-vmax, vmax=vmax, marker=",", linewidths=0, alpha=0.8)
        plt.colorbar(sc, ax=axes[ax_idx], fraction=0.03, pad=0.04, label="SII")
    for ax in axes[n_pairs:]:
        ax.set_visible(False)
    fig.suptitle("Spatial Shapley Interaction Maps (SII order 2)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "interaction_maps.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual figures + summary
    sep = "-" * 52
    print(f"\n{sep}\nMean SII per pair")
    print(sep)
    for i, j in pairs:
        vi, vj = view_names[i], view_names[j]
        vals   = interaction_matrix[:, i, j]
        fig, ax = plt.subplots(figsize=(6, 7))
        _setup_ax(ax, borders, f"SII  {vi} × {vj}", lon_min, lon_max, lat_min, lat_max)
        vmax_p = np.abs(vals).max()
        sc = ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
                        c=vals[idx], cmap="RdBu",
                        vmin=-vmax_p, vmax=vmax_p, marker=",", linewidths=0, alpha=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="SII")
        plt.tight_layout()
        p = out_dir / f"interaction_map_{vi}_{vj}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {vi:<10} x {vj:<10}  mean={vals.mean():+.4f}  |mean|={np.abs(vals).mean():.4f}")
    print(sep)


# ── Scalar SII matrix (from eval.py) ─────────────────────────────────────────

def plot_sii_matrix(
    sii: dict,
    view_names: list,
    out_dir,
    title_suffix: str = "",
    shapley_values: dict = None,
):
    """
    Plot a heatmap of scalar Shapley Interaction Index values.

    One figure per metric. If shapley_values is provided, the diagonal
    shows the order-1 Shapley value φᵢ for each view.

    Args:
        sii:            {metric_name: {(view_i, view_j): value}}
                        Values can be scalars or {"mean": ..., "std": ...}.
        view_names:     Ordered list of view names (defines matrix axes).
        out_dir:        Output directory.
        title_suffix:   Optional string appended to the figure title.
        shapley_values: Optional {metric_name: {view_name: phi}} or
                        {metric_name: {view_name: {"mean": ..., "std": ...}}}.
                        If provided, fills the diagonal with φᵢ.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(view_names)

    for metric_name, pairs in sii.items():
        matrix     = np.full((n, n), np.nan)
        std_matrix = np.full((n, n), np.nan)

        # ── Off-diagonal: SII values ──────────────────────────────────────────
        for i, vi in enumerate(view_names):
            for j, vj in enumerate(view_names):
                if i == j:
                    continue
                key = (vi, vj) if (vi, vj) in pairs else (vj, vi)
                if key not in pairs:
                    continue
                val = pairs[key]
                if isinstance(val, dict):
                    matrix[i, j]     = val["mean"]
                    std_matrix[i, j] = val.get("std", np.nan)
                else:
                    matrix[i, j] = val

        # ── Diagonal: order-1 Shapley values (if provided) ───────────────────
        if shapley_values and metric_name in shapley_values:
            sv = shapley_values[metric_name]
            for i, vi in enumerate(view_names):
                if vi not in sv:
                    continue
                val = sv[vi]
                if isinstance(val, dict):
                    matrix[i, i]     = val["mean"]
                    std_matrix[i, i] = val.get("std", np.nan)
                else:
                    matrix[i, i] = val

        # ── Plot ──────────────────────────────────────────────────────────────
        # Use a shared scale across off-diagonal; diagonal may differ visually
        offdiag = matrix.copy()
        np.fill_diagonal(offdiag, np.nan)
        vmax = np.nanmax(np.abs(offdiag))

        fig, ax = plt.subplots(figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="SII  [diag = φ]",
                     fraction=0.04, pad=0.04, shrink=0.7)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(view_names, fontsize=11)
        ax.set_yticklabels(view_names, fontsize=11)

        title = f"Shapley Interaction Index — {metric_name}"
        if shapley_values:
            title += "  [diag = φ]"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=12)

        for i in range(n):
            for j in range(n):
                if np.isnan(matrix[i, j]):
                    continue
                val = matrix[i, j]
                tc  = "white" if abs(val) > vmax * 0.55 else "#222222"
                ax.text(j, i, f"{val:+.3f}",
                        ha="center", va="center", fontsize=9, color=tc)
                if not np.isnan(std_matrix[i, j]):
                    ax.text(j, i + 0.28, f"±{std_matrix[i,j]:.3f}",
                            ha="center", va="center",
                            fontsize=7, color=tc, alpha=0.7)

        plt.tight_layout()
        p = out_dir / f"sii_matrix_{metric_name}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SII matrix ({metric_name}) saved -> {p}", flush=True)


# ── Shapley values barplot ────────────────────────────────────────────────────

def plot_shapley_barplot(shapley_data: list, view_names: list, out_dir, title_suffix: str = ""):
    """
    Barplot of Shapley values per view, one figure per metric.

    Args:
        shapley_data: List of per-fold row-lists [{metric, view, shapley_value}, ...]
        view_names:   Ordered list of view names.
        out_dir:      Output directory.
        title_suffix: Optional string appended to figure titles.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.DataFrame(s) for s in shapley_data], ignore_index=True)
    COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
              "#59a14f", "#edc948", "#b07aa1", "#9c755f"]

    for metric_name, grp in df.groupby("metric"):
        stats = (grp.groupby("view")["shapley_value"]
                 .agg(mean="mean", std="std")
                 .reindex(view_names)
                 .reset_index())

        fig, ax = plt.subplots(figsize=(max(5, len(view_names) * 1.4), 4))
        x      = np.arange(len(view_names))
        colors = [COLORS[i % len(COLORS)] for i in range(len(view_names))]

        ax.bar(x, stats["mean"], color=colors,
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(x, stats["mean"], yerr=stats["std"],
                    fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=3)

        for xi, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
            sign = "+" if m >= 0 else ""
            offset = s + 0.003 if m >= 0 else -(s + 0.003)
            ax.text(xi, m + offset, f"{sign}{m:.3f}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=9, color="#333333")

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(view_names, fontsize=11)
        ax.set_ylabel("Shapley value φ", fontsize=11)
        title = f"Shapley values — {metric_name}"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        p = out_dir / f"shapley_barplot_{metric_name}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Shapley barplot ({metric_name}) saved -> {p}", flush=True)


# ── SII barplot ───────────────────────────────────────────────────────────────

def plot_sii_barplot(interaction_data: list, out_dir, title_suffix: str = ""):
    """
    Barplot of SII values per view pair, one figure per metric.
    Sorted by absolute value. Green = synergy, red = redundancy.

    Args:
        interaction_data: List of per-fold row-lists [{metric, view_i, view_j, sii_value}, ...]
        out_dir:          Output directory.
        title_suffix:     Optional string appended to figure titles.
    """
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.DataFrame(s) for s in interaction_data], ignore_index=True)
    df["pair"] = df["view_i"] + " × " + df["view_j"]

    for metric_name, grp in df.groupby("metric"):
        stats = (grp.groupby("pair")["sii_value"]
                 .agg(mean="mean", std="std")
                 .reset_index()
                 .sort_values("mean", key=abs, ascending=False))

        n_pairs = len(stats)
        fig, ax = plt.subplots(figsize=(max(5, n_pairs * 1.4), 4))
        x      = np.arange(n_pairs)
        colors = ["#2ca02c" if m >= 0 else "#d62728" for m in stats["mean"]]

        ax.bar(x, stats["mean"], color=colors,
               edgecolor="white", linewidth=0.5, zorder=2, alpha=0.85)
        ax.errorbar(x, stats["mean"], yerr=stats["std"],
                    fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=3)

        for xi, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
            sign = "+" if m >= 0 else ""
            offset = s + 0.002 if m >= 0 else -(s + 0.002)
            ax.text(xi, m + offset, f"{sign}{m:.3f}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=8, color="#333333")

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(stats["pair"], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("SII", fontsize=11)
        title = f"Shapley Interaction Index — {metric_name}"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=12)
        ax.legend(handles=[Patch(facecolor="#2ca02c", label="Synergy (+)"),
                            Patch(facecolor="#d62728", label="Redundancy (−)")],
                  fontsize=9, framealpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        p = out_dir / f"sii_barplot_{metric_name}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SII barplot ({metric_name}) saved -> {p}", flush=True)


# ── Dominant modality proportion by continent ─────────────────────────────────

def plot_dominant_modality_by_continent(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
):
    """
    For each point, find the modality with the highest |Shapley value| (dominant
    modality). Then plot, per continent, the proportion of points where each
    modality is dominant.

    Args:
        df:         DataFrame with columns <view>_shapley per view,
                    plus a 'continent' column.
        view_names: Ordered list of modality names.
        out_dir:    Output directory.
    """
    if "continent" not in df.columns:
        print("WARNING: 'continent' column missing, dominant modality plot skipped.",
              flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols = [f"{v}_shapley" for v in view_names]

    # ── Dominant modality per point ───────────────────────────────────────────
    abs_phi           = df[phi_cols].abs()
    dominant_idx      = abs_phi.values.argmax(axis=1)
    df                = df.copy()
    df["dominant"]    = [view_names[i] for i in dominant_idx]

    # ── Proportion per continent ──────────────────────────────────────────────
    counts = (df.groupby(["continent", "dominant"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=view_names, fill_value=0))
    proportions = counts.div(counts.sum(axis=1), axis=0)

    continents = proportions.index.tolist()
    n_cont     = len(continents)
    n_views    = len(view_names)

    COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
              "#59a14f", "#edc948", "#b07aa1", "#9c755f"]
    color_map = {v: COLORS[i % len(COLORS)] for i, v in enumerate(view_names)}

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_cont * 1.8), 5))
    width   = 0.8 / n_views
    x       = np.arange(n_cont)

    for vi, view in enumerate(view_names):
        offset = (vi - n_views / 2 + 0.5) * width
        vals   = proportions[view].values
        bars   = ax.bar(x + offset, vals, width=width * 0.9,
                        color=color_map[view], label=view,
                        edgecolor="white", linewidth=0.4, zorder=2)

        # Value labels on top of each bar
        for xi, val in zip(x + offset, vals):
            if val > 0.05:
                ax.text(xi, val + 0.01, f"{val:.0%}",
                        ha="center", va="bottom", fontsize=7, color="#333333")

    ns = counts.sum(axis=1).astype(int).tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(continents, ns)],
        fontsize=9,
    )
    ax.set_ylabel("Proportion of points\nwhere modality is dominant", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Dominant modality (highest |φ|) by continent", fontsize=13)
    ax.legend(title="Modality", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    p = out_dir / "dominant_modality_by_continent.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dominant modality plot saved -> {p}", flush=True)

    # ── Also save the proportions as CSV ──────────────────────────────────────
    proportions_out = proportions.copy()
    proportions_out.index.name = "continent"
    proportions_out["n_points"] = counts.sum(axis=1)
    proportions_out.to_csv(out_dir / "dominant_modality_by_continent.csv")


# ── Sensor regime visualizations ──────────────────────────────────────────────

def plot_regime_map(df: pd.DataFrame, out_dir,
                    lon_min, lon_max, lat_min, lat_max):
    """
    Geographic map coloured by sensor regime label.
    One colour per regime, rasterized scatter.
    """
    if "regime" not in df.columns:
        print("WARNING: 'regime' column missing, regime map skipped.", flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    regimes   = sorted(df["regime"].unique())
    n_regimes = len(regimes)
    PALETTE   = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                 "#59a14f", "#edc948", "#b07aa1", "#9c755f"]

    fig, ax = plt.subplots(figsize=(10, 6))
    _setup_ax(ax, borders, f"Sensor regimes (k={n_regimes})",
              lon_min, lon_max, lat_min, lat_max)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))

    for i, regime_id in enumerate(regimes):
        mask  = df["regime"].values[idx] == regime_id
        label = df["regime_label"].values[idx][mask][0] if mask.any() else f"R{regime_id+1}"
        ax.scatter(df["lon"].values[idx][mask],
                   df["lat"].values[idx][mask],
                   c=PALETTE[i % len(PALETTE)],
                   s=0.5, linewidths=0, rasterized=True,
                   label=label)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.85,
              markerscale=6, title="Regime", title_fontsize=9)
    plt.tight_layout()
    p = out_dir / "regime_map.png"
    plt.savefig(p, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Regime map saved -> {p}", flush=True)


def plot_regime_profiles(df: pd.DataFrame, view_names: list, out_dir):
    """
    Grouped barplot showing mean |φ| per view for each regime.
    Each regime is one group of bars — shows which sensors define each regime.
    """
    if "regime" not in df.columns:
        print("WARNING: 'regime' column missing, regime profiles skipped.", flush=True)
        return

    from shap_analysis.regimes import get_regime_profiles

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles  = get_regime_profiles(df, view_names)
    regimes   = profiles.index.tolist()
    n_regimes = len(regimes)
    n_views   = len(view_names)

    PALETTE   = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                 "#59a14f", "#edc948", "#b07aa1", "#9c755f"]
    VIEW_COLORS = [PALETTE[i % len(PALETTE)] for i in range(n_views)]

    fig, ax = plt.subplots(figsize=(max(6, n_regimes * 2), 5))
    width   = 0.8 / n_views
    x       = np.arange(n_regimes)

    for vi, (view, color) in enumerate(zip(view_names, VIEW_COLORS)):
        offset = (vi - n_views / 2 + 0.5) * width
        means  = profiles[f"mean_phi_{view}"].values
        stds   = profiles[f"std_phi_{view}"].values
        ax.bar(x + offset, means, width=width * 0.9,
               color=color, label=view,
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3,
                    linewidth=0.8, zorder=3)

    regime_labels = [df[df["regime"] == r]["regime_label"].iloc[0]
                     for r in regimes]
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("mean |φ|", fontsize=11)
    ax.set_title("Shapley profile per sensor regime", fontsize=13)
    ax.legend(title="Modality", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p = out_dir / "regime_profiles.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Regime profiles saved -> {p}", flush=True)


def plot_regime_continent(df: pd.DataFrame, out_dir):
    """
    Stacked barplot: for each continent, proportion of points per regime.
    Answers: "which regimes dominate in each region of the world?"
    """
    if "regime" not in df.columns or "continent" not in df.columns:
        print("WARNING: 'regime' or 'continent' column missing, skipped.", flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
               "#59a14f", "#edc948", "#b07aa1", "#9c755f"]

    regimes    = sorted(df["regime"].unique())
    regime_lbl = {r: df[df["regime"] == r]["regime_label"].iloc[0] for r in regimes}

    counts = (df.groupby(["continent", "regime"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=regimes, fill_value=0))
    proportions = counts.div(counts.sum(axis=1), axis=0)
    continents  = proportions.index.tolist()
    ns          = counts.sum(axis=1).astype(int).tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(continents) * 1.6), 5))
    bottom = np.zeros(len(continents))

    for i, regime_id in enumerate(regimes):
        vals = proportions[regime_id].values
        ax.bar(np.arange(len(continents)), vals,
               bottom=bottom,
               color=PALETTE[i % len(PALETTE)],
               label=regime_lbl[regime_id],
               edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(np.arange(len(continents)))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(continents, ns)],
                       fontsize=9)
    ax.set_ylabel("Proportion of points", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Sensor regime distribution by continent", fontsize=13)
    ax.legend(title="Regime", fontsize=8, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    p = out_dir / "regime_by_continent.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Regime × continent saved -> {p}", flush=True)


def plot_regime_silhouette(k_analysis: dict, out_dir):
    """
    Elbow + silhouette plot to visualise the optimal k selection.

    Args:
        k_analysis: Output of find_optimal_k().
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks   = k_analysis["k_values"]
    sils = k_analysis["silhouettes"]
    iner = k_analysis["inertias"]
    best = k_analysis["best_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ks, sils, "o-", color="#4e79a7", linewidth=2)
    ax1.axvline(best, color="#e15759", linestyle="--", linewidth=1.2,
                label=f"best k={best}")
    ax1.set_xlabel("k", fontsize=11)
    ax1.set_ylabel("Silhouette score", fontsize=11)
    ax1.set_title("Silhouette (higher = better)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)

    ax2.plot(ks, iner, "o-", color="#f28e2b", linewidth=2)
    ax2.axvline(best, color="#e15759", linestyle="--", linewidth=1.2,
                label=f"best k={best}")
    ax2.set_xlabel("k", fontsize=11)
    ax2.set_ylabel("Inertia", fontsize=11)
    ax2.set_title("Elbow (look for the bend)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    p = out_dir / "regime_k_selection.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"k-selection plot saved -> {p}", flush=True)


# ── Tropical vs non-tropical S2 analysis ─────────────────────────────────────

def plot_tropical_comparison(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
    lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
    signed: bool = False,
):
    """
    Compare Shapley contributions across climate zones.

    Args:
        signed: If False (default), uses |φ| (magnitude).
                If True, uses signed φ — reveals whether a modality
                actively helps (φ>0) or hurts (φ<0) predictions in each zone.

    Produces in out_dir/:
        climate_zone_map.png      (only on first call — same for both modes)
        tropical_barplot.png
        tropical_boxplots.png
        tropical_mannwhitney.png
        tropical_mannwhitney.csv
    """
    from scipy import stats as scipy_stats

    if "climate_zone" not in df.columns:
        print("WARNING: 'climate_zone' column missing, tropical comparison skipped.",
              flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols  = [f"{v}_shapley" for v in view_names]
    zones     = ["Tropical", "Subtropical", "Temperate", "Polar/Boreal"]
    zones     = [z for z in zones if z in df["climate_zone"].unique()]
    phi_label = "φ" if signed else "|φ|"
    mode_tag  = "signed" if signed else "abs"

    def _phi(series):
        return series if signed else series.abs()

    ZONE_COLORS = {
        "Tropical":     "#e15759",
        "Subtropical":  "#f28e2b",
        "Temperate":    "#4e79a7",
        "Polar/Boreal": "#76b7b2",
    }

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ── Mann-Whitney (tropical vs rest) ──────────────────────────────────────
    sep = "-" * 56
    print(f"\n{sep}\n  Tropical vs non-tropical — Mann-Whitney U  ({phi_label})\n{sep}")

    mw_rows      = []
    trop_mask    = df["climate_zone"] == "Tropical"
    nontrop_mask = ~trop_mask

    for view, col in zip(view_names, phi_cols):
        a = _phi(df.loc[trop_mask,    col].dropna()).values
        b = _phi(df.loc[nontrop_mask, col].dropna()).values
        u, p  = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = _stars(p)
        direction = "↓ lower in tropics" if a.mean() < b.mean() else "↑ higher in tropics"
        print(f"  {view:<14}  U={u:.0f}  p={p:.2e}  {stars}  {direction}")
        mw_rows.append({
            "view": view, "U_stat": u, "p_value": p, "significance": stars,
            f"mean_tropical":    a.mean(), f"std_tropical":    a.std(),
            f"mean_nontropical": b.mean(), f"std_nontropical": b.std(),
            "n_tropical": len(a), "n_nontropical": len(b),
        })
    print(sep)

    df_mw = pd.DataFrame(mw_rows)
    df_mw.to_csv(out_dir / "tropical_mannwhitney.csv", index=False)

    # ── 1. Climate zone map (only if not already present) ────────────────────
    borders  = _get_world_borders()
    map_path = out_dir.parent / "climate_zone_map.png"
    if not map_path.exists():
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(df))
        fig, ax = plt.subplots(figsize=(10, 6))
        _setup_ax(ax, borders, "Climate zones", lon_min, lon_max, lat_min, lat_max)
        for zone in zones:
            mask = df["climate_zone"].values[idx] == zone
            ax.scatter(df["lon"].values[idx][mask],
                       df["lat"].values[idx][mask],
                       c=ZONE_COLORS[zone], s=0.5, linewidths=0,
                       rasterized=True, label=f"{zone} (n={mask.sum()})")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85,
                  markerscale=6, title="Climate zone", title_fontsize=9)
        plt.tight_layout()
        plt.savefig(map_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"  -> climate_zone_map.png", flush=True)

    # ── 2. Barplot ────────────────────────────────────────────────────────────
    n_views = len(view_names)
    n_zones = len(zones)
    fig, ax = plt.subplots(figsize=(max(6, n_views * 1.8), 5))
    width   = 0.8 / n_zones
    x       = np.arange(n_views)

    for zi, zone in enumerate(zones):
        sub    = df[df["climate_zone"] == zone]
        means  = [_phi(sub[c].dropna()).mean() for c in phi_cols]
        stds   = [_phi(sub[c].dropna()).std()  for c in phi_cols]
        offset = (zi - n_zones / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=ZONE_COLORS[zone], label=f"{zone} (n={len(sub)})",
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3, linewidth=0.8, zorder=3)

    # Stars only
    for xi, row in enumerate(mw_rows):
        if row["significance"] == "ns":
            continue
        y_vals = [_phi(df[df["climate_zone"] == z][phi_cols[xi]].dropna()).mean() +
                  _phi(df[df["climate_zone"] == z][phi_cols[xi]].dropna()).std()
                  for z in zones]
        ax.text(xi, max(y_vals) + 0.005, row["significance"],
                ha="center", va="bottom", fontsize=11,
                color="#333333", fontweight="bold")

    if signed:
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_ylabel(f"mean {phi_label}", fontsize=11)
    ax.set_title(f"Shapley contributions ({phi_label}) by climate zone\n"
                 "(stars = tropical vs non-tropical significance)", fontsize=11)
    ax.legend(title="Climate zone", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_barplot.png", flush=True)

    # ── 3. Boxplots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_views,
                             figsize=(max(8, n_views * 2.5), 5),
                             sharey=False)
    if n_views == 1:
        axes = [axes]

    for ax, view, col in zip(axes, view_names, phi_cols):
        data   = [_phi(df[df["climate_zone"] == z][col].dropna()).values for z in zones]
        colors = [ZONE_COLORS[z] for z in zones]
        bp = ax.boxplot(data, labels=zones, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        if signed:
            ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(view, fontsize=11)
        ax.set_ylabel(phi_label if ax == axes[0] else "")
        ax.set_xticklabels(zones, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Distribution of {phi_label} by climate zone",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_boxplots.png", flush=True)

    # ── 4. Mann-Whitney figure ────────────────────────────────────────────────
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(max(8, n_views * 2), 4),
                                      gridspec_kw={"wspace": 0.4})
    x = np.arange(n_views)

    ax_u.bar(x, [r["U_stat"] for r in mw_rows],
             color=ZONE_COLORS["Tropical"],
             edgecolor="white", linewidth=0.5, zorder=2)
    ax_u.set_xticks(x)
    ax_u.set_xticklabels(view_names, fontsize=10)
    ax_u.set_ylabel("U statistic", fontsize=10)
    ax_u.set_title(f"Mann-Whitney U ({phi_label})\ntropical vs non-tropical", fontsize=10)
    ax_u.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_u.set_axisbelow(True)

    p_vals     = [r["p_value"] for r in mw_rows]
    bar_colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in p_vals]
    ax_p.bar(x, p_vals, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax_p.axhline(0.05, color="#888888", linewidth=1, linestyle="--", label="p=0.05")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(view_names, fontsize=10)
    ax_p.set_ylabel("p-value", fontsize=10)
    ax_p.set_title("p-values\n(red = significant at 0.05)", fontsize=10)
    ax_p.legend(fontsize=8)
    ax_p.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_p.set_axisbelow(True)

    for xi, row in enumerate(mw_rows):
        direction = "↓" if row["mean_tropical"] < row["mean_nontropical"] else "↑"
        ax_p.text(xi, row["p_value"] + max(p_vals) * 0.02,
                  f"{direction} {row['significance']}",
                  ha="center", va="bottom", fontsize=9, color="#333333")

    fig.suptitle(f"Mann-Whitney test ({phi_label}): tropical vs non-tropical",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_mannwhitney.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_mannwhitney.png", flush=True)
    print(f"Tropical analysis ({mode_tag}) saved -> {out_dir}", flush=True)
    from scipy import stats as scipy_stats

    if "climate_zone" not in df.columns:
        print("WARNING: 'climate_zone' column missing, tropical comparison skipped.",
              flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols = [f"{v}_shapley" for v in view_names]
    zones    = ["Tropical", "Subtropical", "Temperate", "Polar/Boreal"]
    zones    = [z for z in zones if z in df["climate_zone"].unique()]

    ZONE_COLORS = {
        "Tropical":     "#e15759",
        "Subtropical":  "#f28e2b",
        "Temperate":    "#4e79a7",
        "Polar/Boreal": "#76b7b2",
    }

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ── Mann-Whitney (tropical vs rest) ──────────────────────────────────────
    sep = "-" * 56
    print(f"\n{sep}\n  Tropical vs non-tropical — Mann-Whitney U\n{sep}")

    mw_rows      = []
    trop_mask    = df["climate_zone"] == "Tropical"
    nontrop_mask = ~trop_mask

    for view, col in zip(view_names, phi_cols):
        a = _phi(df.loc[trop_mask,    col].dropna()).values
        b = _phi(df.loc[nontrop_mask, col].dropna()).values
        u, p  = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = _stars(p)
        direction = "↓ lower in tropics" if a.mean() < b.mean() else "↑ higher in tropics"
        print(f"  {view:<14}  U={u:.0f}  p={p:.2e}  {stars}  {direction}")
        mw_rows.append({
            "view": view, "U_stat": u, "p_value": p, "significance": stars,
            "mean_tropical":    a.mean(), "std_tropical":    a.std(),
            "mean_nontropical": b.mean(), "std_nontropical": b.std(),
            "n_tropical": len(a), "n_nontropical": len(b),
        })
    print(sep)

    df_mw = pd.DataFrame(mw_rows)
    df_mw.to_csv(out_dir / "tropical_mannwhitney.csv", index=False)

    # ── 1. Climate zone map ───────────────────────────────────────────────────
    borders = _get_world_borders()
    rng     = np.random.default_rng(0)
    idx     = rng.permutation(len(df))

    fig, ax = plt.subplots(figsize=(10, 6))
    _setup_ax(ax, borders, "Climate zones", lon_min, lon_max, lat_min, lat_max)

    for zone in zones:
        mask = df["climate_zone"].values[idx] == zone
        n    = mask.sum()
        ax.scatter(df["lon"].values[idx][mask],
                   df["lat"].values[idx][mask],
                   c=ZONE_COLORS[zone], s=0.5, linewidths=0,
                   rasterized=True, label=f"{zone} (n={n})")

    ax.legend(loc="lower left", fontsize=9, framealpha=0.85,
              markerscale=6, title="Climate zone", title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "climate_zone_map.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  -> climate_zone_map.png", flush=True)

    # ── 2. Barplot — mean |φ| per view × zone (stars, no p-values) ───────────
    n_views = len(view_names)
    n_zones = len(zones)
    fig, ax = plt.subplots(figsize=(max(6, n_views * 1.8), 5))
    width   = 0.8 / n_zones
    x       = np.arange(n_views)

    for zi, zone in enumerate(zones):
        sub    = df[df["climate_zone"] == zone]
        means  = [_phi(sub[c].dropna()).mean() for c in phi_cols]
        stds   = [_phi(sub[c].dropna()).std()  for c in phi_cols]
        offset = (zi - n_zones / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=ZONE_COLORS[zone], label=f"{zone} (n={len(sub)})",
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3, linewidth=0.8, zorder=3)

    # Stars only (tropical vs non-tropical)
    for xi, row in enumerate(mw_rows):
        if row["significance"] == "ns":
            continue
        y_max = max(
            _phi(df[df["climate_zone"] == z][phi_cols[xi]].dropna()).mean() +
            _phi(df[df["climate_zone"] == z][phi_cols[xi]].dropna()).std()
            for z in zones
        )
        ax.text(xi, y_max + 0.005, row["significance"],
                ha="center", va="bottom", fontsize=11, color="#333333",
                fontweight="bold")

    if signed:
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_ylabel(f"mean {phi_label}", fontsize=11)
    ax.set_title(f"Shapley contributions ({phi_label}) by climate zone\n"
                 "(stars = tropical vs non-tropical significance)", fontsize=11)
    ax.legend(title="Climate zone", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_barplot.png", flush=True)

    # ── 3. Boxplots — distribution (no p-values) ─────────────────────────────
    fig, axes = plt.subplots(1, n_views,
                             figsize=(max(8, n_views * 2.5), 5),
                             sharey=False)
    if n_views == 1:
        axes = [axes]

    for ax, view, col in zip(axes, view_names, phi_cols):
        data   = [_phi(df[df["climate_zone"] == z][col].dropna()).values for z in zones]
        colors = [ZONE_COLORS[z] for z in zones]
        bp = ax.boxplot(data, labels=zones, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        if signed:
            ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(view, fontsize=11)
        ax.set_ylabel(phi_label if ax == axes[0] else "")
        ax.set_xticklabels(zones, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Distribution of {phi_label} by climate zone", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_boxplots.png", flush=True)

    # ── 3b. KDE distribution curves — one figure per modality ────────────────
    try:
        from scipy.stats import gaussian_kde
        has_kde = True
    except ImportError:
        has_kde = False
        print("  WARNING: scipy not available, KDE plots skipped.", flush=True)

    if has_kde:
        for view, col in zip(view_names, phi_cols):
            fig, ax = plt.subplots(figsize=(7, 4))

            all_vals = _phi(df[col].dropna()).values
            x_min    = np.percentile(all_vals, 1)
            x_max    = np.percentile(all_vals, 99)
            x_grid   = np.linspace(x_min, x_max, 300)

            for zone in zones:
                vals = _phi(df[df["climate_zone"] == zone][col].dropna()).values
                if len(vals) < 10:
                    continue
                if np.std(vals) == 0: continue
                kde    = gaussian_kde(vals, bw_method="scott")
                color  = ZONE_COLORS[zone]
                n      = len(vals)
                ax.plot(x_grid, kde(x_grid),
                        color=color, linewidth=2,
                        label=f"{zone} (n={n})")
                ax.fill_between(x_grid, kde(x_grid),
                                alpha=0.12, color=color)

            if signed:
                ax.axvline(0, color="#888888", linewidth=0.8,
                           linestyle="--", label="φ = 0")

            ax.set_xlabel(phi_label, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"Distribution of {phi_label} — {view}", fontsize=12)
            ax.legend(fontsize=9, framealpha=0.85)
            ax.grid(linestyle="--", alpha=0.3)
            plt.tight_layout()
            p = out_dir / f"tropical_kde_{view}.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> tropical_kde_{view}.png", flush=True)
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(max(8, n_views * 2), 4),
                                      gridspec_kw={"wspace": 0.4})
    x      = np.arange(n_views)
    colors = [ZONE_COLORS["Tropical"]] * n_views

    # U statistic
    u_vals = [r["U_stat"] for r in mw_rows]
    ax_u.bar(x, u_vals, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax_u.set_xticks(x)
    ax_u.set_xticklabels(view_names, fontsize=10)
    ax_u.set_ylabel("U statistic", fontsize=10)
    ax_u.set_title("Mann-Whitney U\n(tropical vs non-tropical)", fontsize=10)
    ax_u.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_u.set_axisbelow(True)

    # p-values
    p_vals = [r["p_value"] for r in mw_rows]
    bar_colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in p_vals]
    ax_p.bar(x, p_vals, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax_p.axhline(0.05, color="#888888", linewidth=1, linestyle="--", label="p=0.05")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(view_names, fontsize=10)
    ax_p.set_ylabel("p-value", fontsize=10)
    ax_p.set_title("p-values\n(red = significant at 0.05)", fontsize=10)
    ax_p.legend(fontsize=8)
    ax_p.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_p.set_axisbelow(True)

    # Annotate with direction + stars
    for xi, row in enumerate(mw_rows):
        direction = "↓" if row["mean_tropical"] < row["mean_nontropical"] else "↑"
        ax_p.text(xi, row["p_value"] + max(p_vals) * 0.02,
                  f"{direction} {row['significance']}",
                  ha="center", va="bottom", fontsize=9, color="#333333")

    fig.suptitle("Mann-Whitney test: tropical vs non-tropical", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "tropical_mannwhitney.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> tropical_mannwhitney.png", flush=True)

    print(f"Tropical analysis saved -> {out_dir}", flush=True)


# ── SII by climate zone ───────────────────────────────────────────────────────

def plot_sii_by_climate_zone(
    df: pd.DataFrame,
    shapley_matrix: np.ndarray,
    interaction_matrix: np.ndarray,
    view_names: list,
    out_dir,
):
    """
    Compute and plot mean SII per climate zone from already-computed
    per-point spatial arrays (no re-inference needed).

    For each zone, subsets shapley_matrix and interaction_matrix to the
    zone's points, then produces:
        - One heatmap per zone (diagonal = mean φ for that zone)
        - A combined figure with all zones side by side

    Args:
        df:                 DataFrame with 'climate_zone' column, aligned
                            with shapley_matrix rows.
        shapley_matrix:     (n_points, n_views) per-point Shapley values.
        interaction_matrix: (n_points, n_views, n_views) per-point SII.
        view_names:         List of modality names.
        out_dir:            Output directory.
    """
    if "climate_zone" not in df.columns:
        print("WARNING: 'climate_zone' column missing, SII by zone skipped.",
              flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zones   = ["Tropical", "Subtropical", "Temperate", "Polar/Boreal"]
    zones   = [z for z in zones if z in df["climate_zone"].unique()]
    n       = len(view_names)
    n_zones = len(zones)

    # ── Per-zone matrices ─────────────────────────────────────────────────────
    zone_mean_sii = {}
    zone_std_sii  = {}
    zone_mean_phi = {}
    zone_n        = {}

    for zone in zones:
        mask = df["climate_zone"].values == zone
        if mask.sum() == 0:
            continue
        sii_z = interaction_matrix[mask]          # (n_zone, V, V)
        phi_z = shapley_matrix[mask]              # (n_zone, V)
        zone_mean_sii[zone] = sii_z.mean(axis=0)  # (V, V)
        zone_std_sii[zone]  = sii_z.std(axis=0)
        zone_mean_phi[zone] = phi_z.mean(axis=0)  # (V,)
        zone_n[zone]        = mask.sum()

    # Shared colorscale across all zones for comparability
    all_offdiag = np.concatenate([
        zone_mean_sii[z][~np.eye(n, dtype=bool)].flatten()
        for z in zones if z in zone_mean_sii
    ])
    vmax = np.abs(all_offdiag).max()

    # ── Individual heatmaps ───────────────────────────────────────────────────
    for zone in zones:
        if zone not in zone_mean_sii:
            continue
        mean_sii = zone_mean_sii[zone].copy()
        std_sii  = zone_std_sii[zone]
        np.fill_diagonal(mean_sii, zone_mean_phi[zone])

        fig, ax = plt.subplots(figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))
        im = ax.imshow(mean_sii, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="mean(SII)  [diag = mean(φ)]",
                     fraction=0.04, pad=0.04, shrink=0.7)
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(view_names, fontsize=10)
        ax.set_yticklabels(view_names, fontsize=10)
        ax.set_title(f"SII — {zone}  (n={zone_n[zone]})", fontsize=12)

        for i in range(n):
            for j in range(n):
                val = mean_sii[i, j]
                tc  = "white" if abs(val) > vmax * 0.55 else "#222222"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=9, color=tc)
                if i != j:
                    ax.text(j, i + 0.28, f"±{std_sii[i,j]:.3f}",
                            ha="center", va="center",
                            fontsize=7, color=tc, alpha=0.7)

        plt.tight_layout()
        p = out_dir / f"sii_matrix_{zone.replace('/', '_')}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> {p.name}", flush=True)

    # ── Combined figure — all zones side by side ──────────────────────────────
    ncols = min(n_zones, 4)
    nrows = math.ceil(n_zones / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * (n * 1.4 + 1), nrows * (n * 1.2 + 0.5)),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    axes = np.array(axes).flatten()

    for ax_idx, zone in enumerate(zones):
        if zone not in zone_mean_sii:
            continue
        ax       = axes[ax_idx]
        mean_sii = zone_mean_sii[zone].copy()
        np.fill_diagonal(mean_sii, zone_mean_phi[zone])

        im = ax.imshow(mean_sii, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.6,
                     label="SII [diag=φ]")
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(view_names, fontsize=8)
        ax.set_yticklabels(view_names, fontsize=8)
        ax.set_title(f"{zone}\n(n={zone_n[zone]})", fontsize=10)

        for i in range(n):
            for j in range(n):
                val = mean_sii[i, j]
                tc  = "white" if abs(val) > vmax * 0.55 else "#222222"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=tc)

    for ax in axes[n_zones:]:
        ax.set_visible(False)

    fig.suptitle("Shapley Interaction Index by climate zone  [diag = φ]",
                 fontsize=13, y=1.01)
    plt.savefig(out_dir / "sii_by_climate_zone.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> sii_by_climate_zone.png", flush=True)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "-" * 60
    print(f"\n{sep}\n  SII by climate zone — off-diagonal mean\n{sep}")
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    header = f"  {'Pair':<22}" + "".join(f"  {z[:8]:>10}" for z in zones)
    print(header)
    print(f"  {'-'*22}" + "".join(f"  {'':>10}" for _ in zones))
    for i, j in pairs:
        vi, vj = view_names[i], view_names[j]
        row = f"  {vi} × {vj:<{20-len(vi)}}"
        for zone in zones:
            if zone in zone_mean_sii:
                val = zone_mean_sii[zone][i, j]
                row += f"  {val:>+10.4f}"
        print(row)
    print(sep)


# ── φ vs local performance ────────────────────────────────────────────────────

def plot_phi_vs_performance(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
):
    """
    Analyse the relationship between signed Shapley values and local
    classification performance, stratified by true class (crop vs non-crop).

    Uses signed φ so that the direction of contribution is visible:
        φ > 0 : modality pushes toward crop
        φ < 0 : modality pushes toward non-crop

    Produces:
        phi_vs_performance_crop.png      — crop points: TP vs FN
        phi_vs_performance_noncrop.png   — non-crop points: TN vs FP
        phi_vs_performance_combined.png  — mean φ summary barplot
        phi_vs_performance.csv           — Mann-Whitney results
    """
    from scipy import stats as scipy_stats

    required = {"correct", "label"}
    missing  = required - set(df.columns)
    if missing:
        print(f"WARNING: columns {missing} missing, φ vs performance skipped.",
              flush=True)
        return

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols    = [f"{v}_shapley" for v in view_names]
    n_views     = len(view_names)
    label_arr   = df["label"].values.astype(int)
    correct_arr = df["correct"].values.astype(bool)

    COLORS = {
        ("crop",    True):  "#2ca02c",
        ("crop",    False): "#ff7f0e",
        ("noncrop", True):  "#aec7e8",
        ("noncrop", False): "#d62728",
    }
    LABELS = {
        ("crop",    True):  "Crop — correct (TP)",
        ("crop",    False): "Crop — incorrect (FN)",
        ("noncrop", True):  "Non-crop — correct (TN)",
        ("noncrop", False): "Non-crop — incorrect (FP)",
    }

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    def _masks(cls):
        cls_mask = label_arr == (1 if cls == "crop" else 0)
        return cls_mask & correct_arr, cls_mask & ~correct_arr

    # ── Mann-Whitney (signed φ) ───────────────────────────────────────────────
    sep = "-" * 64
    mw_rows = []
    for cls in ("crop", "noncrop"):
        m_ok, m_err = _masks(cls)
        print(f"\n{sep}\n  φ correct vs incorrect — {cls} "
              f"(n_ok={m_ok.sum()}, n_err={m_err.sum()})\n{sep}")
        for view, col in zip(view_names, phi_cols):
            a = df.loc[m_ok,  col].dropna().values
            b = df.loc[m_err, col].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            u, p  = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            stars = _stars(p)
            direction = "↑ higher when correct" if a.mean() > b.mean() \
                        else "↓ lower when correct"
            print(f"  {view:<14}  U={u:.0f}  p={p:.2e}  {stars}  {direction}")
            mw_rows.append({
                "class": cls, "view": view,
                "U_stat": u, "p_value": p, "significance": stars,
                "mean_correct":   a.mean(), "std_correct":   a.std(),
                "mean_incorrect": b.mean(), "std_incorrect": b.std(),
                "n_correct": len(a), "n_incorrect": len(b),
            })
    print(sep)
    pd.DataFrame(mw_rows).to_csv(out_dir / "phi_vs_performance.csv", index=False)

    # ── KDE plots (signed φ) ──────────────────────────────────────────────────
    try:
        from scipy.stats import gaussian_kde
        has_kde = True
    except ImportError:
        has_kde = False

    for cls in ("crop", "noncrop"):
        m_ok, m_err = _masks(cls)
        class_label = "Crop" if cls == "crop" else "Non-crop"

        fig, axes = plt.subplots(1, n_views,
                                 figsize=(max(8, n_views * 3), 4),
                                 sharey=False)
        if n_views == 1:
            axes = [axes]

        for ax, view, col in zip(axes, view_names, phi_cols):
            vals_ok  = df.loc[m_ok,  col].dropna().values
            vals_err = df.loc[m_err, col].dropna().values

            all_vals = np.concatenate([vals_ok, vals_err])
            x_min    = np.percentile(all_vals, 1)
            x_max    = np.percentile(all_vals, 99)
            x_grid   = np.linspace(x_min, x_max, 300)

            if has_kde and len(vals_ok) > 5 and len(vals_err) > 5:
                for vals, key in [(vals_ok, (cls, True)), (vals_err, (cls, False))]:
                    if np.std(vals) == 0:
                        continue
                    kde   = gaussian_kde(vals, bw_method="scott")
                    color = COLORS[key]
                    label = f"{LABELS[key]} (n={len(vals)})"
                    ax.plot(x_grid, kde(x_grid),
                            color=color, linewidth=2.0, label=label)
                    ax.fill_between(x_grid, kde(x_grid),
                                    alpha=0.15, color=color)

            ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
            ax.axvline(vals_ok.mean(),  color=COLORS[(cls, True)],
                       linestyle=":", linewidth=1.2, alpha=0.8)
            ax.axvline(vals_err.mean(), color=COLORS[(cls, False)],
                       linestyle=":", linewidth=1.2, alpha=0.8)

            mw = next((r for r in mw_rows
                       if r["class"] == cls and r["view"] == view), None)
            title = f"{view}  {mw['significance']}" \
                    if mw and mw["significance"] != "ns" else view
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("φ", fontsize=10)
            ax.set_ylabel("Density" if ax == axes[0] else "")
            ax.legend(fontsize=7, framealpha=0.8)
            ax.grid(linestyle="--", alpha=0.3)

        fig.suptitle(f"φ distributions — correct vs incorrect  [{class_label} points]",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"phi_vs_performance_{cls}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> {p.name}", flush=True)

    # ── Combined barplot — mean φ (signed) ───────────────────────────────────
    groups = [
        ("crop",    True,  "Crop — correct (TP)"),
        ("crop",    False, "Crop — incorrect (FN)"),
        ("noncrop", True,  "Non-crop — correct (TN)"),
        ("noncrop", False, "Non-crop — incorrect (FP)"),
    ]
    fig, ax = plt.subplots(figsize=(max(6, n_views * 2), 5))
    width = 0.8 / len(groups)
    x     = np.arange(n_views)

    for gi, (cls, ok, label) in enumerate(groups):
        m_ok, m_err = _masks(cls)
        mask   = m_ok if ok else m_err
        means  = [df.loc[mask, c].mean() for c in phi_cols]
        stds   = [df.loc[mask, c].std()  for c in phi_cols]
        offset = (gi - len(groups) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=COLORS[(cls, ok)], label=f"{label} (n={mask.sum()})",
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=2,
                    linewidth=0.8, zorder=3)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_ylabel("mean φ", fontsize=11)
    ax.set_title("Mean φ per modality — stratified by class and correctness",
                 fontsize=11)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p = out_dir / "phi_vs_performance_combined.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> phi_vs_performance_combined.png", flush=True)
    print(f"φ vs performance saved -> {out_dir}", flush=True)


# ── Subpopulation visualizations ─────────────────────────────────────────────

def plot_subpopulation_profiles(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
):
    """
    Visualise GMM subpopulations discovered on signed Shapley profiles.

    Produces:
        subpop_profiles.png     — mean φ per view per subpopulation (signed barplot)
        subpop_map.png          — geographic map coloured by subpopulation
        subpop_continent.png    — stacked bar: subpop proportion per continent
        subpop_performance.png  — accuracy and crop rate per subpopulation
        subpop_kde_{view}.png   — φ KDE per subpopulation, one figure per view
        subpop_summary.csv
    """
    from shap_analysis.subpopulations import get_subpop_profiles
    from scipy.stats import gaussian_kde

    if "subpop" not in df.columns:
        print("WARNING: 'subpop' column missing, subpopulation plots skipped.",
              flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols  = [f"{v}_shapley" for v in view_names]
    profiles  = get_subpop_profiles(df, view_names)
    subpops   = profiles.index.tolist()
    n_subpops = len(subpops)
    n_views   = len(view_names)

    PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
               "#59a14f", "#edc948", "#b07aa1", "#9c755f"]
    subpop_colors = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(subpops)}
    subpop_labels = {sp: df[df["subpop"] == sp]["subpop_label"].iloc[0]
                     for sp in subpops}

    # ── 1. Profiles barplot (signed φ) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_subpops * 2), 5))
    width   = 0.8 / n_views
    x       = np.arange(n_subpops)

    for vi, view in enumerate(view_names):
        offset = (vi - n_views / 2 + 0.5) * width
        means  = profiles[f"mean_phi_{view}"].values
        stds   = profiles[f"std_phi_{view}"].values
        ax.bar(x + offset, means, width=width * 0.9,
               label=view, edgecolor="white", linewidth=0.4, zorder=2,
               color=PALETTE[vi % len(PALETTE)])
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3,
                    linewidth=0.8, zorder=3)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([subpop_labels[sp] for sp in subpops],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("mean φ", fontsize=11)
    ax.set_title("Signed Shapley profiles per subpopulation", fontsize=13)
    ax.legend(title="Modality", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "subpop_profiles.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> subpop_profiles.png", flush=True)

    # ── 2. Geographic map — small multiples ──────────────────────────────────
    if "lon" in df.columns and "lat" in df.columns:
        borders = _get_world_borders()
        lon_min, lon_max = df["lon"].min() - 1, df["lon"].max() + 1
        lat_min, lat_max = df["lat"].min() - 1, df["lat"].max() + 1

        ncols = 3
        nrows = math.ceil(n_subpops / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 3.5 * nrows),
                                 gridspec_kw={"hspace": 0.35, "wspace": 0.1})
        axes = np.array(axes).flatten()

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(df))

        for ax_idx, sp in enumerate(subpops):
            ax    = axes[ax_idx]
            mask  = df["subpop"].values[idx] == sp
            color = subpop_colors[sp]

            _setup_ax(ax, borders, subpop_labels[sp],
                      lon_min, lon_max, lat_min, lat_max)
            ax.set_title(subpop_labels[sp], fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=7)

            # Background: all other points in light grey
            ax.scatter(df["lon"].values[idx][~mask],
                       df["lat"].values[idx][~mask],
                       c="#dddddd", s=0.3, linewidths=0,
                       rasterized=True, zorder=1)
            # Foreground: this subpopulation in colour
            ax.scatter(df["lon"].values[idx][mask],
                       df["lat"].values[idx][mask],
                       c=color, s=0.5, linewidths=0,
                       rasterized=True, zorder=2)

        for ax in axes[n_subpops:]:
            ax.set_visible(False)

        fig.suptitle(f"Subpopulation geographic distribution (k={n_subpops})",
                     fontsize=13, y=1.01)
        plt.savefig(out_dir / "subpop_map.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> subpop_map.png", flush=True)

    # ── 3. Continent × subpopulation stacked bar ──────────────────────────────
    if "continent" in df.columns:
        counts = (df.groupby(["continent", "subpop"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=subpops, fill_value=0))
        proportions = counts.div(counts.sum(axis=1), axis=0)
        continents  = proportions.index.tolist()
        ns          = counts.sum(axis=1).astype(int).tolist()

        fig, ax = plt.subplots(figsize=(max(8, len(continents) * 1.6), 5))
        bottom = np.zeros(len(continents))
        for sp in subpops:
            vals = proportions[sp].values
            ax.bar(np.arange(len(continents)), vals, bottom=bottom,
                   color=subpop_colors[sp], label=subpop_labels[sp],
                   edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_xticks(np.arange(len(continents)))
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(continents, ns)],
                           fontsize=9)
        ax.set_ylabel("Proportion of points", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title("Subpopulation distribution by continent", fontsize=13)
        ax.legend(title="Subpopulation", fontsize=7, title_fontsize=8,
                  loc="upper right", framealpha=0.85)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / "subpop_continent.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> subpop_continent.png", flush=True)

    # ── 4. Performance per subpopulation ─────────────────────────────────────
    if "correct" in df.columns or "label" in df.columns:
        fig, ax = plt.subplots(figsize=(max(6, n_subpops * 1.8), 4))
        x = np.arange(n_subpops)

        if "correct" in df.columns:
            acc = profiles["accuracy"].values
            ax.bar(x - 0.2, acc, width=0.35,
                   color=[subpop_colors[sp] for sp in subpops],
                   alpha=0.9, label="Accuracy", edgecolor="white", linewidth=0.4)
            for xi, v in enumerate(acc):
                ax.text(xi - 0.2, v + 0.01, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=8)

        if "crop_rate" in profiles.columns:
            cr = profiles["crop_rate"].values
            ax.bar(x + 0.2, cr, width=0.35,
                   color=[subpop_colors[sp] for sp in subpops],
                   alpha=0.45, label="Crop rate", edgecolor="white",
                   linewidth=0.4, hatch="//")
            for xi, v in enumerate(cr):
                ax.text(xi + 0.2, v + 0.01, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([subpop_labels[sp] for sp in subpops],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_title("Accuracy and crop rate per subpopulation", fontsize=12)
        ax.axhline(0.5, color="#888888", linewidth=0.7, linestyle="--")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(out_dir / "subpop_performance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> subpop_performance.png", flush=True)

    # ── 5. KDE per view ───────────────────────────────────────────────────────
    for view, col in zip(view_names, phi_cols):
        fig, ax = plt.subplots(figsize=(7, 4))
        all_vals = df[col].dropna().values
        x_grid   = np.linspace(np.percentile(all_vals, 1),
                               np.percentile(all_vals, 99), 300)

        for sp in subpops:
            vals = df[df["subpop"] == sp][col].dropna().values
            if len(vals) < 10:
                continue
            if np.std(vals) == 0: continue
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid),
                    color=subpop_colors[sp], linewidth=2,
                    label=subpop_labels[sp])
            ax.fill_between(x_grid, kde(x_grid),
                            alpha=0.1, color=subpop_colors[sp])

        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("φ", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"φ distribution per subpopulation — {view}", fontsize=12)
        ax.legend(fontsize=7, framealpha=0.85)
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"subpop_kde_{view}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> subpop_kde_{view}.png", flush=True)

    # ── 6. BIC/silhouette selection plot ──────────────────────────────────────

    # ── CSV summary ───────────────────────────────────────────────────────────
    summary = profiles.copy()
    summary.index.name = "subpop_id"
    summary["label"] = [subpop_labels[sp] for sp in subpops]
    summary.to_csv(out_dir / "subpop_summary.csv")
    print(f"  -> subpop_summary.csv", flush=True)
    print(f"Subpopulation analysis saved -> {out_dir}", flush=True)


def plot_subpop_k_selection(k_analysis: dict, out_dir):
    """BIC + silhouette curves for GMM k selection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks   = k_analysis["k_values"]
    bics = k_analysis["bics"]
    sils = k_analysis["silhouettes"]
    best = k_analysis["recommended_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ks, bics, "o-", color="#4e79a7", linewidth=2)
    ax1.axvline(best, color="#e15759", linestyle="--", linewidth=1.2,
                label=f"recommended k={best}")
    ax1.set_xlabel("k", fontsize=11)
    ax1.set_ylabel("BIC (lower = better)", fontsize=11)
    ax1.set_title("BIC — GMM model selection", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)

    ax2.plot(ks, sils, "o-", color="#f28e2b", linewidth=2)
    ax2.axvline(best, color="#e15759", linestyle="--", linewidth=1.2,
                label=f"recommended k={best}")
    ax2.set_xlabel("k", fontsize=11)
    ax2.set_ylabel("Silhouette (higher = better)", fontsize=11)
    ax2.set_title("Silhouette score", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    p = out_dir / "subpop_k_selection.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> subpop_k_selection.png", flush=True)


# ── φ par décision du modèle ──────────────────────────────────────────────────

def plot_phi_by_predicted_class(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
):
    """
    Compare mean signed φ between:
        - All points (global)
        - Points predicted crop   (pred_score > 0)
        - Points predicted non-crop (pred_score <= 0)

    Answers: "which modalities drive the crop/non-crop decision?"

    Produces:
        phi_by_predicted_class.png   — barplot mean φ × 3 groups
        phi_kde_predicted_class.png  — KDE per modality × 3 groups
        phi_by_predicted_class.csv
    """
    if "pred_score" not in df.columns:
        print("WARNING: 'pred_score' column missing, skipped.", flush=True)
        return

    from scipy.stats import gaussian_kde

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_cols = [f"{v}_shapley" for v in view_names]
    n_views  = len(view_names)

    mask_crop    = df["pred_score"].values > 0
    mask_noncrop = ~mask_crop

    groups = [
        ("Global",         np.ones(len(df), dtype=bool), "#888888"),
        ("Prédit crop",    mask_crop,                     "#2ca02c"),
        ("Prédit non-crop",mask_noncrop,                  "#d62728"),
    ]

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = []
    for group_name, mask, _ in groups:
        sub = df[mask]
        row = {"group": group_name, "n": mask.sum()}
        for view, col in zip(view_names, phi_cols):
            row[f"mean_phi_{view}"] = sub[col].mean()
            row[f"std_phi_{view}"]  = sub[col].std()
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_dir / "phi_by_predicted_class.csv", index=False)

    # Print
    sep = "-" * 64
    print(f"\n{sep}\n  mean φ by predicted class\n{sep}")
    for _, row in df_summary.iterrows():
        vals = "  ".join(
            f"{v}={row[f'mean_phi_{v}']:.3f}" for v in view_names)
        print(f"  {row['group']:<20} (n={int(row['n']):>6})  {vals}")
    print(sep)

    # ── Barplot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_views * 2), 5))
    width   = 0.8 / len(groups)
    x       = np.arange(n_views)

    for gi, (group_name, mask, color) in enumerate(groups):
        sub    = df[mask]
        means  = [sub[c].mean() for c in phi_cols]
        stds   = [sub[c].std()  for c in phi_cols]
        offset = (gi - len(groups) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=color, label=f"{group_name} (n={mask.sum()})",
               edgecolor="white", linewidth=0.4, zorder=2, alpha=0.85)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3,
                    linewidth=0.8, zorder=3)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_ylabel("mean φ", fontsize=11)
    ax.set_title("Shapley contributions selon la décision du modèle\n"
                 "(φ signé — baseline = 0.5)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "phi_by_predicted_class.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> phi_by_predicted_class.png", flush=True)

    # ── KDE par modalité ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_views,
                             figsize=(max(8, n_views * 3), 4),
                             sharey=False)
    if n_views == 1:
        axes = [axes]

    for ax, view, col in zip(axes, view_names, phi_cols):
        all_vals = df[col].dropna().values
        x_grid   = np.linspace(np.percentile(all_vals, 1),
                               np.percentile(all_vals, 99), 300)

        for group_name, mask, color in groups:
            vals = df[mask][col].dropna().values
            if len(vals) < 10:
                continue
            if np.std(vals) == 0:
                continue
            kde   = gaussian_kde(vals, bw_method="scott")
            lw    = 1.5 if group_name == "Global" else 2.0
            ls    = "--" if group_name == "Global" else "-"
            ax.plot(x_grid, kde(x_grid), color=color,
                    linewidth=lw, linestyle=ls,
                    label=f"{group_name} (n={mask.sum()})")
            if group_name != "Global":
                ax.fill_between(x_grid, kde(x_grid),
                                alpha=0.1, color=color)

        ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
        ax.set_xlabel("φ", fontsize=10)
        ax.set_ylabel("Densité" if ax == axes[0] else "")
        ax.set_title(view, fontsize=11)
        ax.legend(fontsize=7, framealpha=0.85)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Distribution de φ selon la décision du modèle",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "phi_kde_predicted_class.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> phi_kde_predicted_class.png", flush=True)
    print(f"φ by predicted class saved -> {out_dir}", flush=True)


# ── Shapley maps split by predicted class ─────────────────────────────────────

def plot_shapley_maps_by_class(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
    lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
    resolution: float = 2.0,
    class_names: dict = None,
):
    """
    Generate Shapley maps (mean |φ|) split by predicted class.
    One combined figure per class showing all modalities.

    Args:
        class_names: optional dict {class_id: label} e.g. {0: "Wheat", 1: "Maize"}
    """
    if "pred_class" not in df.columns:
        print("WARNING: 'pred_class' column missing, skipping per-class maps.", flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    lons = df["lon"].values
    lats = df["lat"].values
    classes = sorted(df["pred_class"].unique())
    n_views = len(view_names)

    def _to_grid(lons_, lats_, values, res):
        lon_edges = np.arange(-180, 180 + res, res)
        lat_edges = np.arange(-90,   90 + res, res)
        n_lon = len(lon_edges) - 1
        n_lat = len(lat_edges) - 1
        lon_idx = np.clip(np.digitize(lons_, lon_edges) - 1, 0, n_lon - 1)
        lat_idx = np.clip(np.digitize(lats_, lat_edges) - 1, 0, n_lat - 1)
        flat    = lon_idx * n_lat + lat_idx
        valid   = ~np.isnan(values)
        gs = np.bincount(flat[valid], weights=values[valid],
                         minlength=n_lon * n_lat).reshape(n_lon, n_lat)
        gc = np.bincount(flat[valid], minlength=n_lon * n_lat).reshape(n_lon, n_lat)
        grid = np.where(gc > 0, gs / gc, np.nan)
        lc  = (lon_edges[:-1] + lon_edges[1:]) / 2
        ltc = (lat_edges[:-1] + lat_edges[1:]) / 2
        return grid, lc, ltc

    # Shared vmax across all classes for comparability
    vmax = max(
        np.nanpercentile(df[f"{v}_shapley"].abs().values, 98)
        for v in view_names
    )

    for cls in classes:
        mask = df["pred_class"].values == cls
        df_cls = df[mask]
        n_cls  = mask.sum()
        label  = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        print(f"  Class {cls} ({label}): {n_cls} points", flush=True)

        ncols = min(n_views, 4)
        nrows = math.ceil(n_views / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4 * nrows),
                                 gridspec_kw={"hspace": 0.3, "wspace": 0.3})
        axes = np.array(axes).flatten()

        for ax_idx, view in enumerate(view_names):
            values = df_cls[f"{view}_shapley"].abs().values
            lons_cls = df_cls["lon"].values
            lats_cls = df_cls["lat"].values
            grid, lc, ltc = _to_grid(lons_cls, lats_cls, values, resolution)
            _setup_ax(axes[ax_idx], borders, view,
                      lon_min, lon_max, lat_min, lat_max)
            pc = axes[ax_idx].pcolormesh(lc, ltc, grid.T,
                                         cmap="plasma", vmin=0, vmax=vmax,
                                         shading="auto")
            plt.colorbar(pc, ax=axes[ax_idx], fraction=0.03,
                         shrink=0.6, pad=0.04, label="|φ|")

        for ax in axes[n_views:]:
            ax.set_visible(False)

        fig.suptitle(f"Shapley values — {label}  (n={n_cls}, {resolution}°×{resolution}° grid)",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        p = out_dir / f"shapley_map_class{cls}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {p.name}", flush=True)

    print(f"Per-class Shapley maps saved -> {out_dir}", flush=True)


# ── KDE de φ signé par classe prédite ────────────────────────────────────────

def plot_shapley_kde_by_class(
    df: pd.DataFrame,
    view_names: list,
    out_dir,
    class_names: dict = None,
):
    """
    Pour chaque modalité, une figure avec une courbe KDE de φ signé par classe prédite.
    Permet de voir si le modèle utilise différemment les modalités selon la classe.
    """
    from scipy.stats import gaussian_kde

    if "pred_class" not in df.columns:
        print("WARNING: 'pred_class' column missing, KDE by class skipped.", flush=True)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(df["pred_class"].unique())
    n_classes = len(classes)

    PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2",
               "#59a14f","#edc948","#b07aa1","#9c755f",
               "#bab0ac","#d37295"]

    for view in view_names:
        col = f"{view}_shapley"
        if col not in df.columns:
            continue

        all_vals = df[col].dropna().values
        x_min = np.percentile(all_vals, 1)
        x_max = np.percentile(all_vals, 99)
        x_grid = np.linspace(x_min, x_max, 400)

        fig, ax = plt.subplots(figsize=(9, 5))

        for cls in classes:
            mask = df["pred_class"].values == cls
            vals = df[mask][col].dropna().values
            if len(vals) < 10 or np.std(vals) == 0:
                continue
            label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
            color = PALETTE[cls % len(PALETTE)]
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2,
                    label=f"{label} (n={mask.sum()})")
            ax.fill_between(x_grid, kde(x_grid), alpha=0.08, color=color)

        ax.axvline(0, color="#888888", linewidth=0.9, linestyle="--", label="φ = 0")
        ax.set_xlabel("φ (signé)", fontsize=11)
        ax.set_ylabel("Densité", fontsize=11)
        ax.set_title(f"Distribution de φ par classe prédite — {view}", fontsize=12)
        ax.legend(fontsize=8, framealpha=0.85,
                  ncol=max(1, n_classes // 6),
                  loc="upper right")
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        p = out_dir / f"kde_by_class_{view}.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> {p.name}", flush=True)

    # Figure combinée — toutes les modalités, une ligne par classe
    n_views = len(view_names)
    fig, axes = plt.subplots(n_classes, n_views,
                             figsize=(3.5 * n_views, 2.5 * n_classes),
                             sharex="col", sharey=False)
    if n_classes == 1:
        axes = axes[np.newaxis, :]
    if n_views == 1:
        axes = axes[:, np.newaxis]

    for ri, cls in enumerate(classes):
        mask  = df["pred_class"].values == cls
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        color = PALETTE[cls % len(PALETTE)]
        for ci, view in enumerate(view_names):
            ax  = axes[ri, ci]
            col = f"{view}_shapley"
            vals = df[mask][col].dropna().values
            if len(vals) >= 10 and np.std(vals) > 0:
                all_v  = df[col].dropna().values
                x_grid = np.linspace(np.percentile(all_v, 1),
                                     np.percentile(all_v, 99), 300)
                kde = gaussian_kde(vals, bw_method="scott")
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5)
                ax.fill_between(x_grid, kde(x_grid), alpha=0.2, color=color)
            ax.axvline(0, color="#888888", linewidth=0.7, linestyle="--")
            ax.set_yticks([])
            ax.grid(linestyle="--", alpha=0.2)
            if ri == 0:
                ax.set_title(view, fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0,
                              labelpad=60, va="center")

    fig.suptitle("φ signé par classe prédite × modalité", fontsize=12, y=1.01)
    plt.tight_layout()
    p = out_dir / "kde_by_class_combined.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> kde_by_class_combined.png", flush=True)
    print(f"KDE by class saved -> {out_dir}", flush=True)