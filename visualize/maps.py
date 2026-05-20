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


def _scatter_discrete(ax, lons, lats, values, groups, s=10, alpha=0.6):
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
    ax.scatter(lons[idx], lats[idx], c=colors, s=s, linewidths=0, alpha=alpha)
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
                      lon_min, lon_max, lat_min, lat_max):
    """
    One map per modality showing |Shapley value| as a continuous colormap.
    Also saves a combined figure with all modalities.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    borders = _get_world_borders()

    vmax = max(df[f"{v}_shapley"].abs().max() for v in view_names)
    n    = len(view_names)

    # Combined figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"hspace": 0.1, "wspace": 0.35})
    axes = axes.flatten()
    for ax, view in zip(axes, view_names):
        values = df[f"{view}_shapley"].values
        _setup_ax(ax, borders, view, lon_min, lon_max, lat_min, lat_max)
        sc = ax.scatter(df["lon"], df["lat"], c=np.abs(values),
                        cmap="YlOrRd", vmin=0, vmax=vmax, s=12, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.03, shrink=0.6, pad=0.04, label="|Shapley value|")
    fig.suptitle("Shapley values per modality", fontsize=15, y=0.90)
    plt.tight_layout()
    p = out_dir / "shapley_map.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined map saved -> {p}", flush=True)

    # Individual figures
    for view in view_names:
        values = df[f"{view}_shapley"].values
        vmax_v = np.abs(values).max()
        fig, ax = plt.subplots(figsize=(6, 7))
        _setup_ax(ax, borders, f"Shapley - {view}", lon_min, lon_max, lat_min, lat_max)
        sc = ax.scatter(df["lon"], df["lat"], c=np.abs(values),
                        cmap="YlOrRd", vmin=0, vmax=vmax_v, s=14, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.03, shrink=0.6, pad=0.04, label="|Shapley value|")
        plt.tight_layout()
        p = out_dir / f"shapley_map_{view}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
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
                         cmap="YlGn", vmin=0.0, vmax=1.0, s=10, linewidths=0, alpha=0.7)
    plt.colorbar(sc, ax=ax_pred, fraction=0.03, pad=0.04, label="P(crop)")

    fig.suptitle("Ground truth vs predicted score", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "gt_pred_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Separate GT
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, borders, "Ground truth (crop / non-crop)", lon_min, lon_max, lat_min, lat_max)
    handles = _scatter_discrete(ax, df["lon"].values, df["lat"].values,
                                 df["label"].values.astype(int), GT_COLORS)
    ax.legend(handles=handles, loc="lower left", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "gt_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Separate pred score
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, borders, "Predicted score (all views)", lon_min, lon_max, lat_min, lat_max)
    sc = ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
                    c=df["pred_score"].values[idx],
                    cmap="YlGn", vmin=0.0, vmax=1.0, s=10, linewidths=0, alpha=0.7)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="P(crop)")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_map.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(out_dir / "correct_map.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(out_dir / f"{tag}_map.png", dpi=150, bbox_inches="tight")
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
               c=colors, s=10, linewidths=0, alpha=0.65)
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
    plt.savefig(p, dpi=150, bbox_inches="tight")
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
    plt.savefig(p, dpi=150, bbox_inches="tight")
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
    plt.savefig(p, dpi=150, bbox_inches="tight")
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
                                  vmin=-vmax, vmax=vmax, s=10, linewidths=0, alpha=0.7)
        plt.colorbar(sc, ax=axes[ax_idx], fraction=0.03, pad=0.04, label="SII")
    for ax in axes[n_pairs:]:
        ax.set_visible(False)
    fig.suptitle("Spatial Shapley Interaction Maps (SII order 2)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "interaction_maps.png", dpi=150, bbox_inches="tight")
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
                        vmin=-vmax_p, vmax=vmax_p, s=10, linewidths=0, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="SII")
        plt.tight_layout()
        p = out_dir / f"interaction_map_{vi}_{vj}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {vi:<10} x {vj:<10}  mean={vals.mean():+.4f}  |mean|={np.abs(vals).mean():.4f}")
    print(sep)