"""
eval.py
-------
Charge un checkpoint sauvegarde (run/fold) et evalue les metriques sur le test set associe.

Usage :
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0 --shapley --save_predictions preds/run0_fold0.npy
    python eval.py -s config/dsensdp_ex.yaml -w path/to/model_run0_fold0.pt --shapley
"""

import argparse
import copy
import itertools
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


print("Imports standards OK", flush=True)

try:
    from code.training.learn_pipeline import MultiFusion_train
    from code.training.learn_pipeline import build_model
    from code.training.utils import assign_multifusion_name, output_name, assign_labels_weights
    from code.datasets.views_structure import Dataset_MultiView
    from code.datasets.utils import create_dataloader, load_structure
    print("Imports projet OK", flush=True)
except Exception:
    print(f"\nERREUR lors des imports projet :\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


# ==============================================================================
#  Helpers
# ==============================================================================

def log(msg: str):
    print(msg, flush=True)


def _shapley_values(view_names, v_dict):
    n = len(view_names)
    shapley = {}
    v_empty = v_dict.get(frozenset(), 0)  # v(vide)

    for view in view_names:
        others = [v for v in view_names if v != view]
        phi = v_empty / n
        for size in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, size):
                S = frozenset(S_tuple)
                s = len(S)
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi += weight * (v_dict[S | {view}] - v_dict[S])
        shapley[view] = phi
    return shapley


def _compute_metrics(y_true, y_pred_proba, task_type):
    task = (task_type or "").lower()
    if task == "multilabel":
        y_pred = (y_pred_proba > 0.5).astype(int)
        print("Warning: using 0.5 threshold for multilabel classification, consider tuning this threshold for better performance.")
    else:
        y_pred = y_pred_proba.argmax(axis=-1)
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro",    zero_division=0),
    }


def _get_random_baseline_metrics(y_true, task_type):
    # Calcule les metriques attendues pour un classifieur aleatoire
    task = (task_type or "").lower()

    if task == "multilabel":
        return {
            "accuracy":        0.5,
            "f1_macro":        0.5,
            "f1_weighted":     0.5,
            "precision_macro": 0.5,
            "recall_macro":    0.5,
        }

    unique_classes = np.unique(y_true)

    if len(unique_classes) == 2:
        p_pos = np.mean(y_true)
        p_neg = 1 - p_pos

        tp = p_pos * p_pos
        fn = p_pos * p_neg
        fp = p_neg * p_pos
        tn = p_neg * p_neg

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = tp + tn

        return {
            "accuracy":        accuracy,
            "f1_macro":        f1,
            "f1_weighted":     f1,
            "precision_macro": precision,
            "recall_macro":    recall,
        }

    elif len(unique_classes) > 2:
        n_classes  = len(unique_classes)
        random_acc = 1.0 / n_classes
        return {
            "accuracy":        random_acc,
            "f1_macro":        random_acc,
            "f1_weighted":     random_acc,
            "precision_macro": random_acc,
            "recall_macro":    random_acc,
        }

    else:
        return {
            "accuracy":        0.5,
            "f1_macro":        0.5,
            "f1_weighted":     0.5,
            "precision_macro": 0.5,
            "recall_macro":    0.5,
        }


def _print_metrics(metrics: dict, title: str = "Metriques"):
    sep = "-" * 48
    log(f"\n{sep}")
    log(f"  {title}")
    log(sep)
    for k, v in metrics.items():
        log(f"  {k:<24}: {v:.4f}")
    log(sep)


# ==============================================================================
#  Reconstruction du modele via 0-epoch
# ==============================================================================

def _build_model(checkpoint: dict, config: dict):
    import copy
    cfg = copy.deepcopy(config)
    input_dir = cfg["input_dir_folder"]
    data_name = cfg["data_name"]

    log(f"  Chargement des donnees pour construire l'architecture ({data_name})...")
    data_views = load_structure(input_dir, data_name, load_memory=cfg.get("load_memory", False))
    data_views.load_stats(input_dir, data_name)
    data_views.set_additional_info(**cfg["experiment"].get("preprocess", {}))

    if cfg.get("task_type", "").lower() in ["classification", "multilabel"]:
        assign_labels_weights(cfg, data_views)

    log("  Construction du modele (sans entrainement)...")
    model = build_model(data_views, **cfg)

    log("  Chargement du state_dict...")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log("  State-dict charge.")
    return model


# ==============================================================================
#  Chargement du test set
# ==============================================================================

def _load_test_data(checkpoint: dict, config: dict):
    input_dir = config["input_dir_folder"]
    data_name = config["data_name"]

    if "train" in data_name:
        test_name = data_name.replace("train", "test")
        log(f"  Split fixe -> chargement de : {test_name}")
        data_te = load_structure(input_dir, test_name, load_memory=config.get("load_memory", False))
        data_te.load_stats(input_dir, data_name)
    else:
        indices_file = checkpoint.get("test_indices_file")
        if not indices_file or not Path(indices_file).exists():
            raise FileNotFoundError(
                f"Indices de test introuvables ({indices_file}). "
                "Relancez l'entrainement avec save_test_set=True."
            )
        log(f"  K-fold -> indices depuis : {indices_file}")
        test_indices = np.load(indices_file, allow_pickle=True)
        data_te = load_structure(input_dir, data_name, load_memory=config.get("load_memory", False))
        data_te.load_stats(input_dir, data_name)
        data_te.set_val_mask(test_indices)
        data_te.set_data_mode(train=False)

    data_te.set_additional_info(**config["experiment"].get("preprocess", {}))
    log(f"  {len(data_te)} echantillons de test.")
    return data_te


# ==============================================================================
#  Sauvegarde CSV
# ==============================================================================

def _save_results_csv(
    out_dir: Path,
    run_id: int,
    fold_id: int,
    metrics: dict,
    shapley_per_metric: dict = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_run{run_id}_fold{fold_id}.csv"
    row = {"run_id": run_id, "fold_id": fold_id, **metrics}
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    log(f"\nMetriques sauvegardees -> {metrics_path}")

    if shapley_per_metric:
        shapley_rows = []
        for metric_name, sv in shapley_per_metric.items():
            for view, phi in sv.items():
                shapley_rows.append({
                    "run_id":        run_id,
                    "fold_id":       fold_id,
                    "metric":        metric_name,
                    "view":          view,
                    "shapley_value": phi,
                })
        shapley_path = out_dir / f"shapley_run{run_id}_fold{fold_id}.csv"
        pd.DataFrame(shapley_rows).to_csv(shapley_path, index=False)
        log(f"Valeurs de Shapley sauvegardees -> {shapley_path}")


# ==============================================================================
#  Point d'entree
# ==============================================================================

def run_inference(args):

    log(f"\nLecture du config : {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    log("  Config charge.")

    run_id  = args.run_id
    fold_id = args.fold_id

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
        weights_dir = config.get(
            "weights_dir",
            f"{config['output_dir_folder']}/weights/{config['data_name']}",
        )
        weights_path = Path(weights_dir) / method_name / f"model_run{run_id}_fold{fold_id}.pt"
        log(f"  Chemin checkpoint reconstruit : {weights_path}")

    if not weights_path.exists():
        log(f"\nERREUR checkpoint introuvable : {weights_path}")
        log("  Verifiez --weights_file ou que save_weights=True etait active a l'entrainement.")
        sys.exit(1)

    log(f"\nChargement du checkpoint : {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)
    log(
        f"   run_id     = {checkpoint.get('run_id',  run_id)}\n"
        f"   fold_id    = {checkpoint.get('fold_id', fold_id)}\n"
        f"   epoch      = {checkpoint.get('epoch',  '?')}\n"
        f"   best_score = {checkpoint.get('best_score', '?')}"
    )

    log("\nReconstruction du modele...")
    method = _build_model(checkpoint, ckpt_config)

    log("\nChargement du test set...")
    data_te = _load_test_data(checkpoint, ckpt_config)

    BS        = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type = ckpt_config.get("task_type", "")
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    log(f"\nInference (batch_size={BS})...")
    t0 = time.time()
    with torch.no_grad():
        outputs = method.transform(
            create_dataloader(data_te, batch_size=BS, train=False),
            out_norm=output_name(task_type),
            not_return_repre=True,
        )
    log(f"  Termine en {time.time() - t0:.2f}s")

    y_true = data_te.get_all_labels()
    y_pred = outputs["prediction"]

    metrics = _compute_metrics(y_true, y_pred, task_type)
    _print_metrics(metrics, title=f"Test - run {run_id} / fold {fold_id}")

    shapley_per_metric = None
    if args.shapley:
        view_names  = ckpt_config["experiment"]["preprocess"]["view_names"]
        metric_keys = list(metrics.keys())
        log(f"\nCalcul des valeurs de Shapley sur {len(view_names)} vues...")
        log(f"  Vues : {view_names}")

        baseline = _get_random_baseline_metrics(y_true, task_type)
        v_dict = {
            frozenset():           baseline,
            frozenset(view_names): metrics,
        }

        n_subsets = sum(math.comb(len(view_names), s) for s in range(1, len(view_names)))
        done = 0
        for size in range(1, len(view_names)):
            for subset_tuple in itertools.combinations(view_names, size):
                subset_list  = list(subset_tuple)
                subset_label = "+".join(subset_list)
                args_fwd = {
                    "inference_views": subset_list,
                    "missing_method":  method.missing_method,
                }
                with torch.no_grad():
                    out_sub = method.transform(
                        create_dataloader(data_te, batch_size=BS, train=False),
                        out_norm=output_name(task_type),
                        args_forward=args_fwd,
                        perc_forward=1.0,
                        not_return_repre=True,
                    )
                sub_metrics = _compute_metrics(y_true, out_sub["prediction"], task_type)
                v_dict[frozenset(subset_list)] = sub_metrics
                done += 1
                log(f"  [{done}/{n_subsets}] {subset_label} -> f1_macro={sub_metrics['f1_macro']:.4f}")

        shapley_per_metric = {}
        sep = "-" * 48
        for metric_name in metric_keys:
            v_scalar = {s: v_dict[s][metric_name] for s in v_dict}
            sv = _shapley_values(view_names, v_scalar)
            shapley_per_metric[metric_name] = sv
            log(f"\n  Shapley ({metric_name}) :")
            log(sep)
            for view, phi in sorted(sv.items(), key=lambda x: -abs(x[1])):
                sign = "+" if phi >= 0 else ""
                log(f"    {view:<20} {sign}{phi:.4f}")
            log(sep)

    out_dir = Path(args.save_predictions).parent if args.save_predictions else Path("preds")

    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, y_pred)
        log(f"\nPredictions (.npy) -> {pred_path}")

    _save_results_csv(out_dir, run_id, fold_id, metrics, shapley_per_metric)

    log("\nTermine.")
    return metrics


# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference et evaluation d'un checkpoint MultiFusion."
    )
    p.add_argument("--settings_file", "-s", required=True,
                   help="Fichier de config YAML (identique a celui utilise a l'entrainement).")
    p.add_argument("--weights_file",  "-w", default=None,
                   help="Chemin explicite vers le .pt (optionnel).")
    p.add_argument("--run_id",  "-r", type=int, default=0)
    p.add_argument("--fold_id", "-f", type=int, default=0)
    p.add_argument("--batch_size", "-b", type=int, default=None,
                   help="Batch size pour l'inference (defaut : valeur du config).")
    p.add_argument("--shapley", action="store_true",
                   help="Calcule et sauvegarde les valeurs de Shapley par vue.")
    p.add_argument("--save_predictions", default=None, metavar="PATH",
                   help="Chemin pour sauvegarder les predictions brutes (.npy). "
                        "Le dossier parent recoit aussi les CSV de resultats.")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_inference(args)
    except SystemExit:
        raise
    except Exception:
        log("\nERREUR non geree :\n")
        log(traceback.format_exc())
        sys.exit(1)
