"""
infer_france.py
---------------
Pour chaque point du test set situe en France, calcule les valeurs de Shapley
de chaque modalite [S1, S2, DEM, weather] en utilisant le logit predit comme
valeur du jeu, puis genere une carte par modalite.

Usage :
    python infer_france.py -s config/dsensdp_ex.yaml
    python infer_france.py -s config/dsensdp_ex.yaml -w path/to/model_run0_fold0.pt
    python infer_france.py -s config/dsensdp_ex.yaml -w /home/casado/DsensDp/DSensDp/res_out/weights/Dec_avg-SD-ignore-Plus/model_run0_fold0.pt
"""

import argparse
import itertools
import math
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

print("Imports standards OK", flush=True)

try:
    from eval import _build_model, _load_test_data, log
    from code.training.utils import assign_multifusion_name, output_name
    from code.datasets.utils import create_dataloader
    print("Imports projet OK", flush=True)
except Exception:
    print(f"\nERREUR lors des imports :\n{traceback.format_exc()}", flush=True)
    sys.exit(1)

# Bounding box France metropolitaine
# FRANCE_LON_MIN = -5.5
# FRANCE_LON_MAX =  9.5
# FRANCE_LAT_MIN = 41.0
# FRANCE_LAT_MAX = 51.5
FRANCE_LON_MIN = -180
FRANCE_LON_MAX =  180
FRANCE_LAT_MIN = -90
FRANCE_LAT_MAX = 90

NC_PATH = "/home/casado/DsensDp/data/out/cropharvest_binary.nc"


# ==============================================================================
#  Coordonnees
# ==============================================================================

def build_coords(data_te, nc_path=NC_PATH):
    log(f"Chargement des coordonnees depuis {nc_path}...")
    ds        = xr.open_dataset(nc_path)
    ids_nc    = ds["identifier"].values
    coords_nc = ds["coords"].values          # (n_total, 2) -> [lon, lat]
    id2idx    = {id_: i for i, id_ in enumerate(ids_nc)}

    ids_te  = data_te.get_all_identifiers()
    missing = [id_ for id_ in ids_te if id_ not in id2idx]
    if missing:
        raise KeyError(f"{len(missing)} identifiers absents du .nc : {missing[:5]}")

    coords = np.array([coords_nc[id2idx[id_]] for id_ in ids_te])
    log(f"  Coordonnees alignees : {coords.shape}")
    return coords


# ==============================================================================
#  Inference par sous-ensemble de vues
# ==============================================================================

def _infer_subset(method, data_te, subset_list, batch_size, task_type):
    """
    Lance l'inference en n'utilisant que les vues de subset_list.
    Retourne les predictions du modele : (n_samples,) ou (n_samples, n_classes).
    """
    with torch.no_grad():
        out = method.transform(
            create_dataloader(data_te, batch_size=batch_size, train=False),
            out_norm=output_name(task_type),
            args_forward={
                "inference_views": subset_list,
                "missing_method":  method.missing_method,
            },
            perc_forward=1.0,
            not_return_repre=True,
        )
    return out["prediction"]


# ==============================================================================
#  Shapley par point
# ==============================================================================

def compute_shapley(method, data_te, bbox_idx, batch_size, task_type, view_names):
    """
    Calcule les valeurs de Shapley pour chaque point de bbox_idx.

    v(S, x)  = p_crop predit par le modele pour le point x
               quand seules les vues S sont disponibles.
    v(vide)  = proportion de crop dans le test set (classifieur aleatoire).

    Retourne shapley_matrix : (n_bbox, n_views)
    """
    n_views = len(view_names)
    n_bbox  = len(bbox_idx)

    # Baseline : proportion de crop dans le test set entier
    labels  = data_te.get_all_labels()
    v_empty = float(np.mean(labels == 1))
    log(f"  Baseline v(vide) = {v_empty:.4f} (proportion crop dans le test set)")

    # Tous les sous-ensembles non vides, du plus petit au plus grand
    all_subsets = []
    for size in range(1, n_views + 1):
        for combo in itertools.combinations(view_names, size):
            all_subsets.append(list(combo))

    n_subsets = len(all_subsets)
    log(f"  {n_subsets} passes d'inference ({n_views} vues, 2^{n_views}-1 sous-ensembles)...")

    v_dict    = {frozenset(): np.full(n_bbox, v_empty, dtype=np.float64)}
    ref_class = None

    for i, subset_list in enumerate(all_subsets):
        label       = "+".join(subset_list)
        logits_all  = _infer_subset(method, data_te, subset_list, batch_size, task_type)
        logits_bbox = logits_all[bbox_idx]

        if logits_bbox.ndim > 1:
            if ref_class is None:
                ref_class = np.ones(n_bbox, dtype=int)  # toujours classe 1 (crop)
            logits_bbox = logits_bbox[np.arange(n_bbox), ref_class]

        v_dict[frozenset(subset_list)] = logits_bbox.astype(np.float64)
        log(f"    [{i+1}/{n_subsets}] {label}")

    log("  Calcul Shapley par point...")
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

    # Recupere les predictions avec toutes les vues (grand ensemble)
    full_preds = v_dict[frozenset(view_names)]

    return shapley_matrix, full_preds, v_dict


# ==============================================================================
#  Carte ground truth + predicted score
# ==============================================================================

def _scatter_discrete(ax, lons, lats, values, groups, s=10, alpha=0.6):
    """
    Trace un scatter discret en melangeant l'ordre des points pour eviter
    que la derniere categorie ecrase les autres.

    groups : liste de (valeur, label, couleur)
    Retourne les handles pour ax.legend().
    """
    rng    = np.random.default_rng(0)
    idx    = rng.permutation(len(lons))
    lons_s = lons[idx]
    lats_s = lats[idx]
    vals_s = values[idx]

    color_map = {val: color for val, _, color in groups}
    colors    = np.array([color_map[v] for v in vals_s])

    ax.scatter(lons_s, lats_s, c=colors, s=s, linewidths=0, alpha=alpha)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=7, label=name)
        for _, name, color in groups
    ]
    return handles



# ==============================================================================
#  Interactions de Shapley (SII ordre 2)
# ==============================================================================

def compute_shapley_interactions(v_dict, view_names, n_bbox):
    """
    Calcule le Shapley Interaction Index (SII) d'ordre 2 pour toutes les paires,
    en reutilisant v_dict deja calcule par compute_shapley.

    SII(i,j) = sum_{S subset N\\{i,j}} w(|S|) * delta_ij(S)
    delta_ij(S) = v(S u {i,j}) - v(S u {i}) - v(S u {j}) + v(S)
    w(s)        = s! * (n-s-2)! / (n-1)!

    Retourne interaction_matrix : (n_bbox, n_views, n_views), symetrique,
    diagonale = valeurs de Shapley d'ordre 1 (convention SII diagonale = SV).
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
            interaction_matrix[:, j, i] = phi_ij   # symetrique

    log("  SII calcule pour toutes les paires.")
    return interaction_matrix



# ==============================================================================
#  Attribution des continents (bbox approximative)
# ==============================================================================

# Ordre important : les bbox se chevauchent, on teste du plus specifique au plus large
_CONTINENT_BBOX = [
    ("Antarctique",     -180, -90,  180, -60),
    ("Oceanie",          110, -50,  180,  10),
    ("Amerique du Nord", -170,  15,  -50,  85),
    ("Amerique du Sud",  -82, -57,  -34,  15),
    ("Europe",           -25,  35,   45,  72),
    ("Afrique",          -20, -35,   55,  38),
    ("Asie",              25,  -5,  180,  80),
]

def assign_continents(lons, lats):
    """
    Attribue un continent a chaque point (lon, lat) par bounding box.
    Retourne un array de strings, "Inconnu" si aucune bbox ne correspond.

    Essaie d'abord geopandas+naturalearth (point-in-polygon exact),
    se rabat sur les bbox approximatives si indisponible.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from pathlib import Path as _Path

        _NE_SHP = _Path.home() / ".cache" / "naturalearth" / "ne_110m_admin_0_countries.shp"
        if not _NE_SHP.exists():
            raise FileNotFoundError(
                f"{_NE_SHP} introuvable. Telechargez le avec :\n"
                "  python -c \"import urllib.request,zipfile,io,pathlib; "
                "data=urllib.request.urlopen('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip').read(); "
                "dest=pathlib.Path.home()/'.cache'/'naturalearth'; dest.mkdir(parents=True,exist_ok=True); "
                "zipfile.ZipFile(io.BytesIO(data)).extractall(dest)\""
            )

        world = gpd.read_file(_NE_SHP)[["CONTINENT", "geometry"]].rename(
            columns={"CONTINENT": "continent"}
        )
        world = world[world["continent"].notna()].copy()
        pts   = gpd.GeoDataFrame(
            {"geometry": [Point(lo, la) for lo, la in zip(lons, lats)]},
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, world, how="left", predicate="within")
        continents = joined["continent"].fillna("Inconnu").values
        log("  Continents assignes via ne_110m_admin_0_countries (point-in-polygon).")
        return continents
    except Exception as e:
        log(f"  geopandas indisponible ou shapefile manquant ({e}), utilisation des bounding boxes.")

    log("  geopandas indisponible, utilisation des bounding boxes approximatives.")
    continents = np.full(len(lons), "Inconnu", dtype=object)
    for name, lon_min, lat_min, lon_max, lat_max in _CONTINENT_BBOX:
        mask = (
            (lons >= lon_min) & (lons <= lon_max) &
            (lats >= lat_min) & (lats <= lat_max) &
            (continents == "Inconnu")
        )
        continents[mask] = name
    return continents


# ==============================================================================
#  Statistiques groupees des valeurs de Shapley
# ==============================================================================


def plot_continent_map(df, out_dir, lon_min, lon_max, lat_min, lat_max):
    """
    Carte des points colores par continent, sauvegardee dans out_dir/stats/.
    """
    if "continent" not in df.columns:
        log("AVERTISSEMENT : colonne 'continent' absente, carte continents ignoree.")
        return

    out_dir = Path(out_dir) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    borders = _get_world_borders()

    def _setup(ax, title):
        _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max)
        ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=13)

    # Palette — une couleur fixe par continent pour coherence avec les CSV
    PALETTE = {
        "Europe":           "#4e79a7",
        "Asie":             "#f28e2b",
        "Afrique":          "#e15759",
        "Amerique du Nord": "#76b7b2",
        "Amerique du Sud":  "#59a14f",
        "Oceanie":          "#edc948",
        "Antarctique":      "#b07aa1",
        "Inconnu":          "#bab0ac",
        # noms anglais si geopandas est utilise
        "Europe":           "#4e79a7",
        "Asia":             "#f28e2b",
        "Africa":           "#e15759",
        "North America":    "#76b7b2",
        "South America":    "#59a14f",
        "Oceania":          "#edc948",
        "Antarctica":       "#b07aa1",
        "Unknown":          "#bab0ac",
    }

    continents_present = sorted(df["continent"].unique())
    groups = [
        (c, PALETTE.get(c, "#cccccc"))
        for c in continents_present
    ]

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    lons_s = df["lon"].values[idx]
    lats_s = df["lat"].values[idx]
    cont_s = df["continent"].values[idx]

    color_map = {c: col for c, col in groups}
    colors    = np.array([color_map[c] for c in cont_s])

    fig, ax = plt.subplots(figsize=(10, 6))
    _setup(ax, f"Points par continent (n={len(df)})")
    ax.scatter(lons_s, lats_s, c=colors, s=10, linewidths=0, alpha=0.65)

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
    log(f"Carte continents sauvegardee -> {p}")

def compute_stats(df, view_names, out_dir):
    """
    Calcule mean(|phi_i|) pour chaque modalite selon trois decoupages :
      1. Global (tous les points)
      2. Par continent
      3. Par prediction correcte / incorrecte (si colonnes disponibles)

    Sauvegarde un CSV par decoupage dans out_dir/stats/.
    Affiche aussi un resume dans les logs.
    """
    stats_dir = Path(out_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    phi_cols = [f"{v}_shapley" for v in view_names]
    sep      = "-" * 60

    def _group_stats(df_sub, group_col, group_label):
        rows = []
        for group_val, gdf in df_sub.groupby(group_col):
            n = len(gdf)
            row = {group_col: group_val, "n_points": n}
            for col, view in zip(phi_cols, view_names):
                vals = gdf[col].abs()
                row[f"|phi|_mean_{view}"] = vals.mean()
                row[f"|phi|_std_{view}"]  = vals.std()
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 1. Global
    # ------------------------------------------------------------------
    log(f"\n{sep}")
    log(f"  mean(|phi|) GLOBAL  ({len(df)} points)")
    log(sep)
    global_rows = []
    for col, view in zip(phi_cols, view_names):
        vals = df[col].abs()
        log(f"    {view:<12}  mean={vals.mean():.4f}  std={vals.std():.4f}")
        global_rows.append({"modalite": view,
                             "|phi|_mean": vals.mean(),
                             "|phi|_std":  vals.std(),
                             "n_points":   len(df)})
    pd.DataFrame(global_rows).to_csv(stats_dir / "stats_global.csv", index=False)
    log(f"  -> {stats_dir / 'stats_global.csv'}")

    # ------------------------------------------------------------------
    # 2. Par continent
    # ------------------------------------------------------------------
    if "continent" in df.columns:
        log(f"\n{sep}")
        log("  mean(|phi|) PAR CONTINENT")
        log(sep)
        df_cont = _group_stats(df, "continent", "continent")
        for _, row in df_cont.iterrows():
            vals_str = "  ".join(
                f"{v}={row[f'|phi|_mean_{v}']:.4f}" for v in view_names
            )
            log(f"    {row['continent']:<22} (n={int(row['n_points']):>5})  {vals_str}")
        df_cont.to_csv(stats_dir / "stats_par_continent.csv", index=False)
        log(f"  -> {stats_dir / 'stats_par_continent.csv'}")
    else:
        log("  AVERTISSEMENT : colonne 'continent' absente, stats par continent ignorees.")

    # ------------------------------------------------------------------
    # 3. Par prediction correcte / incorrecte
    # ------------------------------------------------------------------
    if "correct" in df.columns:
        log(f"\n{sep}")
        log("  mean(|phi|) PAR PREDICTION (correct / incorrect)")
        log(sep)
        label_map  = {True: "Correct", False: "Incorrect"}
        df_corr    = df.copy()
        df_corr["prediction"] = df_corr["correct"].map(label_map)
        df_pred = _group_stats(df_corr, "prediction", "prediction")
        for _, row in df_pred.iterrows():
            vals_str = "  ".join(
                f"{v}={row[f'|phi|_mean_{v}']:.4f}" for v in view_names
            )
            log(f"    {row['prediction']:<12} (n={int(row['n_points']):>5})  {vals_str}")
        df_pred.to_csv(stats_dir / "stats_par_prediction.csv", index=False)
        log(f"  -> {stats_dir / 'stats_par_prediction.csv'}")

        # Croise continent x prediction si les deux sont disponibles
        if "continent" in df.columns:
            df_cross = df_corr.copy()
            df_cross["group"] = df_cross["continent"] + " / " + df_cross["prediction"]
            df_cx = _group_stats(df_cross, "group", "group")
            df_cx.to_csv(stats_dir / "stats_continent_x_prediction.csv", index=False)
            log(f"  -> {stats_dir / 'stats_continent_x_prediction.csv'}")
    else:
        log("  AVERTISSEMENT : colonne 'correct' absente, stats par prediction ignorees.")

    log(sep)

def plot_interaction_graph(shapley_matrix, interaction_matrix, view_names, out_dir):
    """
    Graphe d'interaction circulaire (moyenne sur tous les points de la bbox) :
      - Noeuds : modalites, taille proportionnelle a mean|phi_i|
      - Aretes : mean SII(i,j), epaisseur = |valeur|,
                 Bleu = interaction positive (synergie),
                 rouge = interaction negative (redondance)
      - Labels d'arete : valeur moyenne du SII
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_views      = len(view_names)
    mean_phi     = np.abs(shapley_matrix).mean(axis=0)          # (n_views,)
    mean_sii     = interaction_matrix.mean(axis=0)              # (n_views, n_views)
    abs_sii      = np.abs(mean_sii)

    # Positions en cercle
    angles = [2 * math.pi * k / n_views for k in range(n_views)]
    pos    = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angles)}

    # Normalisation taille des noeuds
    node_scale = 3000
    node_sizes = node_scale * mean_phi / (mean_phi.max() + 1e-12)

    # Normalisation epaisseur des aretes
    edge_max  = abs_sii.max() + 1e-12
    lw_scale  = 12

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- aretes ---
    for i in range(n_views):
        for j in range(i + 1, n_views):
            val  = mean_sii[i, j]
            lw   = lw_scale * abs(val) / edge_max
            color = "#2ca02c" if val >= 0 else "#d62728"
            xi, yi = pos[i]
            xj, yj = pos[j]
            ax.plot([xi, xj], [yi, yj], color=color, lw=lw, alpha=0.75, zorder=1)
            # label au milieu de l'arete
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            ax.text(mx, my, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # --- noeuds ---
    for i, name in enumerate(view_names):
        x, y = pos[i]
        ax.scatter(x, y, s=node_sizes[i], color="#4e79a7", zorder=2,
                   edgecolors="white", linewidths=1.5)
        offset = 0.18
        ax.text(x * (1 + offset), y * (1 + offset), name,
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x * (1 + offset * 2.2), y * (1 + offset * 2.2),
                f"|φ|={mean_phi[i]:.3f}",
                ha="center", va="center", fontsize=9, color="#666666")

    ax.set_title("Graphe d'interactions de Shapley (SII ordre 2)",
                 fontsize=13)

    plt.tight_layout()
    p = out_dir / "interaction_graph.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Graphe d'interaction sauvegarde -> {p}")

def plot_interaction_matrix(shapley_matrix, interaction_matrix, view_names, out_dir):
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n        = len(view_names)
    mean_sii = interaction_matrix.mean(axis=0)   # (n_views, n_views)
    std_sii  = interaction_matrix.std(axis=0)

    # Diagonale = valeurs de Shapley moyennes (convention SII)
    mean_phi = shapley_matrix.mean(axis=0)        # (n_views,)
    np.fill_diagonal(mean_sii, mean_phi)

    vmax = np.abs(mean_sii).max()

    fig, ax = plt.subplots(figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))
    im = ax.imshow(mean_sii, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="mean(SII)  [diag = mean(φ)]",
                 fraction=0.04, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(view_names, fontsize=11)
    ax.set_yticklabels(view_names, fontsize=11)
    ax.set_title(
        f"Matrice d'interaction Shapley (SII ordre 2)",
        fontsize=13,
    )

    for i in range(n):
        for j in range(n):
            val = mean_sii[i, j]
            text_color = "white" if abs(val) > vmax * 0.55 else "#222222"
            ax.text(j, i, f"{val:+.3f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color, zorder=2)
            if i != j:
                ax.text(j, i + 0.28, f"±{std_sii[i,j]:.3f}",
                        ha="center", va="center",
                        fontsize=7, color=text_color, alpha=0.7, zorder=2)

    plt.tight_layout()
    p = out_dir / "interaction_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Matrice SII sauvegardee -> {p}")


def plot_interaction_maps(df, interaction_matrix, view_names, out_dir,
                          lon_min, lon_max, lat_min, lat_max):
    """
    Carte spatiale du SII pour chaque paire de modalites (triangle superieur).
    Colormap divergente centree sur 0 : bleu = synergie, rouge = redondance.
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    borders  = _get_world_borders()
    n_views  = len(view_names)
    n_pairs  = n_views * (n_views - 1) // 2
    pairs    = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]

    def _setup(ax, title):
        _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max)
        ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=11)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))

    # vmax commun a toutes les paires (echelle coherente)
    vmax = max(np.abs(interaction_matrix[:, i, j]).max() for i, j in pairs)

    # --- figure combinee ---
    ncols = min(3, n_pairs)
    nrows = math.ceil(n_pairs / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.35})
    axes = np.array(axes).flatten()

    for ax_idx, (i, j) in enumerate(pairs):
        vi, vj = view_names[i], view_names[j]
        vals   = interaction_matrix[:, i, j]
        ax     = axes[ax_idx]
        _setup(ax, f"SII  {vi} × {vj}")
        sc = ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
                        c=vals[idx], cmap="RdBu",
                        vmin=-vmax, vmax=vmax,
                        s=10, linewidths=0, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="SII")

    # masque les axes vides
    for ax in axes[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle("Cartes spatiales des interactions de Shapley (SII ordre 2)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    p = out_dir / "interaction_maps.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Cartes SII sauvegardees -> {p}")

    # --- figures separees par paire ---
    for i, j in pairs:
        vi, vj = view_names[i], view_names[j]
        vals   = interaction_matrix[:, i, j]
        vmax_p = np.abs(vals).max()
        fig, ax = plt.subplots(figsize=(6, 7))
        _setup(ax, f"SII  {vi} × {vj}")
        sc = ax.scatter(df["lon"].values[idx], df["lat"].values[idx],
                        c=vals[idx], cmap="RdBu",
                        vmin=-vmax_p, vmax=vmax_p,
                        s=10, linewidths=0, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="SII")
        plt.tight_layout()
        p = out_dir / f"interaction_map_{vi}_{vj}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  -> {p}")

    # resume numerique
    sep = "-" * 52
    log(f"\n{sep}")
    log("SII moyen par paire (moyenne sur la bbox)")
    log(sep)
    for i, j in pairs:
        vi, vj = view_names[i], view_names[j]
        vals   = interaction_matrix[:, i, j]
        log(f"  {vi:<10} x {vj:<10}  mean={vals.mean():+.4f}  |mean|={np.abs(vals).mean():.4f}")
    log(sep)


def plot_gt_pred_maps(df, out_dir, lon_min, lon_max, lat_min, lat_max):
    """
    Genere deux cartes cote a cote :
      - Gauche  : ground truth (0 = non-crop, 1 = crop), colormap discret
      - Droite  : score predit par le modele (toutes vues), colormap continu [0,1]

    Suppose que df contient les colonnes 'label' et 'pred_score'.
    """
    if "label" not in df.columns or "pred_score" not in df.columns:
        log("AVERTISSEMENT : colonnes 'label' ou 'pred_score' absentes, carte GT/pred ignoree.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    borders = _get_world_borders()

    def _setup_ax(ax, title):
        _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max)
        ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=13)

    # --- figure combinee ---
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"wspace": 0.35})

    # Ground truth (discret : 0 / 1)
    _setup_ax(ax_gt, "Ground truth (crop / non-crop)")
    GT_COLORS = [(0, "Non-crop", "#d9534f"), (1, "Crop", "#5cb85c")]
    handles = _scatter_discrete(ax_gt,
                                df["lon"].values, df["lat"].values,
                                df["label"].values.astype(int),
                                GT_COLORS)
    ax_gt.legend(handles=handles, loc="lower left", framealpha=0.8)

    # Predicted score (continu [0,1])
    _setup_ax(ax_pred, "Score predit (toutes vues)")
    rng_pred  = np.random.default_rng(0)
    idx_pred  = rng_pred.permutation(len(df))
    sc_pred = ax_pred.scatter(df["lon"].values[idx_pred], df["lat"].values[idx_pred],
                              c=df["pred_score"].values[idx_pred],
                              cmap="YlGn",
                              vmin=0.0, vmax=1.0,
                              s=10, linewidths=0, alpha=0.7)
    plt.colorbar(sc_pred, ax=ax_pred, fraction=0.03, pad=0.04,
                 label="P(crop)")

    fig.suptitle("Ground truth vs score predit", fontsize=15, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "gt_pred_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Carte GT/pred sauvegardee -> {out_path}")

    # --- figure separee GT ---
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, "Ground truth (crop / non-crop)")
    handles = _scatter_discrete(ax,
                                df["lon"].values, df["lat"].values,
                                df["label"].values.astype(int),
                                GT_COLORS)
    ax.legend(handles=handles, loc="lower left", framealpha=0.8)
    plt.tight_layout()
    p = out_dir / "gt_map.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  -> {p}")

    # --- figure separee pred score (continu -> colorbar) ---
    fig, ax = plt.subplots(figsize=(6, 7))
    _setup_ax(ax, "Score predit (toutes vues)")
    idx_s = np.random.default_rng(0).permutation(len(df))
    sc = ax.scatter(df["lon"].values[idx_s], df["lat"].values[idx_s],
                    c=df["pred_score"].values[idx_s],
                    cmap="YlGn", vmin=0.0, vmax=1.0,
                    s=10, linewidths=0, alpha=0.7)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="P(crop)")
    plt.tight_layout()
    p = out_dir / "pred_map.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  -> {p}")

    # --- carte correct / incorrect ---
    pred_class = (df["pred_score"].values >= 0.5).astype(int)
    correct    = (pred_class == df["label"].values.astype(int))

    # Distingue 4 cas : TP, TN, FP, FN
    label_arr = df["label"].values.astype(int)
    outcome   = np.where(
        correct & (label_arr == 1), 0,   # TP
        np.where(
        correct & (label_arr == 0), 1,   # TN
        np.where(
        ~correct & (label_arr == 0), 2,  # FP (predit crop, vrai non-crop)
        3,                               # FN (predit non-crop, vrai crop)
    )))

    # figure combinee correct/incorrect + detail TP/TN/FP/FN
    fig, (ax_bin, ax_detail) = plt.subplots(1, 2, figsize=(14, 6),
                                             gridspec_kw={"wspace": 0.35})

    # panneau gauche : correct (vert) vs incorrect (rouge)
    _setup_ax(ax_bin, f"Correct vs incorrect (seuil 0.5)\nacc = {correct.mean():.3f}")
    BIN_COLORS = [(0, "Incorrect", "#d62728"), (1, "Correct", "#2ca02c")]
    h_bin = _scatter_discrete(ax_bin,
                              df["lon"].values, df["lat"].values,
                              correct.astype(int), BIN_COLORS)
    ax_bin.legend(handles=h_bin, loc="lower left", framealpha=0.8)

    # panneau droit : detail TP/TN/FP/FN
    _setup_ax(ax_detail, "Detail des erreurs (TP / TN / FP / FN)")
    ERR_COLORS = [(0, "TP (crop correct)",    "#2ca02c"),
                  (1, "TN (non-crop correct)", "#aec7e8"),
                  (2, "FP (fausse alarme)",    "#d62728"),
                  (3, "FN (manque)",           "#ff7f0e")]
    h_err = _scatter_discrete(ax_detail,
                              df["lon"].values, df["lat"].values,
                              outcome, ERR_COLORS)
    ax_detail.legend(handles=h_err, loc="lower left", framealpha=0.8)

    fig.suptitle("Performance de classification", fontsize=15, y=1.01)
    plt.tight_layout()
    p = out_dir / "correct_map.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Carte correct/incorrect sauvegardee -> {p}")

    # figures separees
    for tag, title, groups, arr in [
        ("correct_bin",
         f"Correct vs incorrect (seuil 0.5) — acc={correct.mean():.3f}",
         BIN_COLORS, correct.astype(int)),
        ("correct_detail",
         "Detail des erreurs (TP / TN / FP / FN)",
         ERR_COLORS, outcome),
    ]:
        fig, ax = plt.subplots(figsize=(6, 7))
        _setup_ax(ax, title)
        handles = _scatter_discrete(ax,
                                    df["lon"].values, df["lat"].values,
                                    arr, groups)
        ax.legend(handles=handles, loc="lower left", framealpha=0.8)
        plt.tight_layout()
        p = out_dir / f"{tag}_map.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  -> {p}")


# ==============================================================================
#  Carte par modalite
# ==============================================================================

def _get_world_borders():
    """
    Charge les frontieres des pays.
    Essaie cartopy, puis geopandas/naturalearth, sinon None.
    """
    # Option 1 : cartopy (le plus fiable, pas de telechargement)
    try:
        import cartopy.feature as cfeature
        return ("cartopy", cfeature.BORDERS, cfeature.COASTLINE)
    except ImportError:
        pass

    # Option 2 : geopandas avec naturalearth embarque dans le package
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
        world = gpd.read_file(_NE_SHP)
        return ("geopandas", world)
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
        world = borders[1]
        world.boundary.plot(ax=ax, color="black", linewidth=0.5)


def plot_shapley_maps(df, view_names, out_dir, lon_min, lon_max, lat_min, lat_max):
    """
    Genere une carte par modalite.
    Chaque point est colore selon sa valeur de Shapley (colormap divergente).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    borders = _get_world_borders()

    def _draw(ax):
        _draw_borders(ax, borders, lon_min, lon_max, lat_min, lat_max)
        ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    n_views = len(view_names)

    # figure combinee
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"hspace": 0.1, "wspace": 0.35})#
    axes = axes.flatten()

    vmax = max(df[f"{view}_shapley"].abs().max() for view in view_names)

    for ax, view in zip(axes, view_names):
        col    = f"{view}_shapley"
        values = df[col].values
        _draw(ax)
        sc = ax.scatter(df["lon"], df["lat"], c=np.abs(values), cmap="YlOrRd",
                        vmin=0, vmax=vmax, s=12, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="|Shapley value|")
        ax.set_title(view, fontsize=13)

    fig.suptitle("Valeurs de Shapley par modalite", fontsize=15, y=0.90)
    plt.tight_layout()
    out_path = out_dir / "shapley_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Carte combinee sauvegardee -> {out_path}")

    # figures par modalite
    for view in view_names:
        col    = f"{view}_shapley"
        values = df[col].values
        vmax   = np.abs(values).max()

        fig, ax = plt.subplots(figsize=(6, 7))
        _draw(ax)
        sc = ax.scatter(df["lon"], df["lat"], c=np.abs(values), cmap="YlOrRd",
                        vmin=0, vmax=vmax, s=14, linewidths=0)
        plt.colorbar(sc, ax=ax, label="|Shapley value|")
        ax.set_title(f"Shapley - {view}", fontsize=13)
        plt.tight_layout()

        p = out_dir / f"shapley_map_{view}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  -> {p}")


# ==============================================================================
#  Main
# ==============================================================================

def main(args):
    log(f"\nLecture du config : {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.safe_load(f)
    log("  Config charge.")

    if args.weights_file:
        weights_path = Path(args.weights_file)
    else:
        if "loss_args" not in config["training"]:
            config["training"]["loss_args"] = {}
        task = config.get("task_type", "").lower()
        if task == "classification":
            config["training"]["loss_args"].setdefault("name", "ce")
        elif task == "multilabel":
            config["training"]["loss_args"].setdefault("name", "bce")
        method_name = assign_multifusion_name(
            config["training"], config["method"],
            more_info_str=config.get("additional_method_name", ""),
        )
        weights_dir  = config.get(
            "weights_dir",
            f"{config['output_dir_folder']}/weights/{config['data_name']}",
        )
        weights_path = (Path(weights_dir) / method_name
                        / f"model_run{args.run_id}_fold{args.fold_id}.pt")
        log(f"  Chemin checkpoint reconstruit : {weights_path}")

    if not weights_path.exists():
        log(f"\nERREUR checkpoint introuvable : {weights_path}")
        sys.exit(1)

    log(f"\nChargement du checkpoint : {weights_path}")
    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)

    log("\nReconstruction du modele...")
    method  = _build_model(checkpoint, ckpt_config)

    log("\nChargement du test set...")
    data_te = _load_test_data(checkpoint, ckpt_config)

    coords = build_coords(data_te, nc_path=args.nc_path)

    lons = coords[:, 0]
    lats = coords[:, 1]
    mask = (
        (lons >= args.lon_min) & (lons <= args.lon_max) &
        (lats >= args.lat_min) & (lats <= args.lat_max)
    )
    bbox_idx = np.where(mask)[0]
    n_bbox   = len(bbox_idx)

    if n_bbox == 0:
        log("Aucun point dans la bounding box France.")
        sys.exit(0)

    log(f"\n{n_bbox} points en France (sur {len(data_te)} total).")

    batch_size = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type  = ckpt_config.get("task_type", "")
    view_names = ckpt_config["experiment"]["preprocess"]["view_names"]
    log(f"  Vues utilisees : {view_names}")
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    log("\nCalcul des valeurs de Shapley...")
    t0                       = time.time()
    shapley_matrix, full_preds, v_dict = compute_shapley(
        method, data_te, bbox_idx, batch_size, task_type, view_names
    )
    log(f"  Termine en {time.time() - t0:.1f}s")

    ids_bbox = data_te.get_all_identifiers()[bbox_idx]
    df_dict  = {
        "sample_idx": bbox_idx,
        "identifier": ids_bbox,
        "lon":        lons[bbox_idx],
        "lat":        lats[bbox_idx],
        "pred_score": full_preds,          # score P(crop) avec toutes les vues
    }
    for j, view in enumerate(view_names):
        df_dict[f"{view}_shapley"] = shapley_matrix[:, j]

    try:
        labels = data_te.get_all_labels()[bbox_idx]
        df_dict["label"] = labels if labels.ndim == 1 else labels[:, 0]
    except Exception:
        pass

    df = pd.DataFrame(df_dict)

    csv_path = Path(args.out_dir) / "shapley_france.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log(f"\nCSV sauvegarde -> {csv_path}")

    # Export geospatial
    if args.geo_format:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            gdf = gpd.GeoDataFrame(
                df,
                geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
                crs="EPSG:4326",
            )
            if args.geo_format == "gpkg":
                geo_path = Path(args.out_dir) / "shapley_france.gpkg"
                gdf.to_file(geo_path, driver="GPKG")
            elif args.geo_format == "geojson":
                geo_path = Path(args.out_dir) / "shapley_france.geojson"
                gdf.to_file(geo_path, driver="GeoJSON")
            log(f"Export geospatial sauvegarde -> {geo_path}")
        except ImportError:
            log("AVERTISSEMENT : geopandas/shapely non disponible, export geospatial ignore.")

    # ------------------------------------------------------------------
    # Sous-dossiers de sortie
    # ------------------------------------------------------------------
    out_root      = Path(args.out_dir)
    dir_gt_pred   = out_root / "maps_gt_pred"
    dir_shapley   = out_root / "maps_shapley"
    dir_interact  = out_root / "maps_interactions"
    for d in (dir_gt_pred, dir_shapley, dir_interact):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Enrichissement du DataFrame : continent + correct
    # ------------------------------------------------------------------
    log("\nAttribution des continents...")
    df["continent"] = assign_continents(df["lon"].values, df["lat"].values)

    if "label" in df.columns and "pred_score" in df.columns:
        pred_class    = (df["pred_score"].values >= 0.5).astype(int)
        df["correct"] = pred_class == df["label"].values.astype(int)

    # ------------------------------------------------------------------
    # Interactions de Shapley (SII ordre 2)
    # ------------------------------------------------------------------
    log("\nCalcul des interactions de Shapley (SII ordre 2)...")
    interaction_matrix = compute_shapley_interactions(v_dict, view_names, n_bbox)

    for i, vi in enumerate(view_names):
        for j, vj in enumerate(view_names):
            if j > i:
                df[f"sii_{vi}_{vj}"] = interaction_matrix[:, i, j]

    # ------------------------------------------------------------------
    # CSV final (toutes colonnes)
    # ------------------------------------------------------------------
    df.to_csv(csv_path, index=False)
    log(f"\nCSV final sauvegarde -> {csv_path}")

    # ------------------------------------------------------------------
    # Statistiques groupees
    # ------------------------------------------------------------------
    log("\nCalcul des statistiques groupees...")
    compute_stats(df, view_names, out_dir=args.out_dir)
    plot_continent_map(df, out_dir=args.out_dir,
                       lon_min=args.lon_min, lon_max=args.lon_max,
                       lat_min=args.lat_min, lat_max=args.lat_max)

    # ------------------------------------------------------------------
    # Cartes
    # ------------------------------------------------------------------
    log("\nGeneration des cartes GT / predicted score...")
    plot_gt_pred_maps(df, out_dir=dir_gt_pred,
                      lon_min=args.lon_min, lon_max=args.lon_max,
                      lat_min=args.lat_min, lat_max=args.lat_max)

    log("\nGeneration des cartes Shapley...")
    plot_shapley_maps(df, view_names, out_dir=dir_shapley,
                      lon_min=args.lon_min, lon_max=args.lon_max,
                      lat_min=args.lat_min, lat_max=args.lat_max)

    log("\nGeneration des graphes d'interaction...")
    plot_interaction_graph(shapley_matrix, interaction_matrix, view_names,
                           out_dir=dir_interact)
    plot_interaction_maps(df, interaction_matrix, view_names,
                          out_dir=dir_interact,
                          lon_min=args.lon_min, lon_max=args.lon_max,
                          lat_min=args.lat_min, lat_max=args.lat_max)
    plot_interaction_matrix(shapley_matrix, interaction_matrix, view_names, out_dir=dir_interact)

    log("\nTermine.")
    log(f"\nStructure de sortie :")
    log(f"  {out_root}/")
    log(f"  ├── shapley_france.csv")
    log(f"  ├── maps_gt_pred/")
    log(f"  ├── maps_shapley/")
    log(f"  ├── maps_interactions/")
    log(f"  └── stats/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Shapley par point sur le test set France."
    )
    p.add_argument("--settings_file", "-s", required=True)
    p.add_argument("--weights_file",  "-w", default=None)
    p.add_argument("--run_id",  "-r", type=int, default=0)
    p.add_argument("--fold_id", "-f", type=int, default=0)
    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--nc_path", default=NC_PATH)
    p.add_argument("--out_dir", "-o", default="preds/france",
                   help="Dossier de sortie pour le CSV et les cartes (defaut: preds/france)")
    p.add_argument("--lon_min", type=float, default=FRANCE_LON_MIN)
    p.add_argument("--lon_max", type=float, default=FRANCE_LON_MAX)
    p.add_argument("--lat_min", type=float, default=FRANCE_LAT_MIN)
    p.add_argument("--lat_max", type=float, default=FRANCE_LAT_MAX)
    p.add_argument("--geo_format", choices=["gpkg", "geojson"], default=None,
                   help="Format d'export geospatial (gpkg ou geojson, optionnel)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        main(parse_args())
    except SystemExit:
        raise
    except Exception:
        log("\nERREUR non geree :\n")
        log(traceback.format_exc())
        sys.exit(1)