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
FRANCE_LON_MIN = -5.5
FRANCE_LON_MAX =  9.5
FRANCE_LAT_MIN = 41.0
FRANCE_LAT_MAX = 51.5

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

    return shapley_matrix


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
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
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

    # --- figure combinee ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"hspace": 0.1, "wspace": 0.35})#
    axes = axes.flatten()

    for ax, view in zip(axes, view_names):
        col    = f"{view}_shapley"
        values = df[col].values
        vmax   = np.abs(values).max()
        _draw(ax)
        sc = ax.scatter(df["lon"], df["lat"], c=values, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=12, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label="Shapley value")
        ax.set_title(view, fontsize=13)

    fig.suptitle("Valeurs de Shapley par modalite", fontsize=15, y=0.90)
    plt.tight_layout()
    out_path = out_dir / "shapley_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Carte combinee sauvegardee -> {out_path}")

    # --- une figure par modalite ---
    for view in view_names:
        col    = f"{view}_shapley"
        values = df[col].values
        vmax   = np.abs(values).max()

        fig, ax = plt.subplots(figsize=(6, 7))
        _draw(ax)
        sc = ax.scatter(df["lon"], df["lat"], c=values, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=14, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Shapley value")
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

    # Filtrage France
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
    t0             = time.time()
    shapley_matrix = compute_shapley(method, data_te, bbox_idx, batch_size, task_type, view_names)
    log(f"  Termine en {time.time() - t0:.1f}s")

    # Construction du DataFrame
    ids_bbox = data_te.get_all_identifiers()[bbox_idx]
    df_dict  = {
        "sample_idx": bbox_idx,
        "identifier": ids_bbox,
        "lon":        lons[bbox_idx],
        "lat":        lats[bbox_idx],
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

    sep = "-" * 48
    log(f"\n{sep}")
    log(f"moyenne de valeur absolue de shapley sur la France ({n_bbox} points)")
    log(sep)
    for view in view_names:
        col  = f"{view}_shapley"
        mean = df[col].abs().mean()
        std  = df[col].abs().std()
        log(f"  |{view:<14}| {mean:.4f}  (+/- {std:.4f})")
    log(sep)

    log("\nGeneration des cartes...")
    plot_shapley_maps(df, view_names, out_dir=args.out_dir,
                      lon_min=args.lon_min, lon_max=args.lon_max,
                      lat_min=args.lat_min, lat_max=args.lat_max)

    log("\nTermine.")


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
