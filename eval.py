"""
eval.py
-------
Load a saved checkpoint (run/fold) and evaluate metrics on the associated test set.

Usage:
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0 --shapley --save_predictions preds/run0_fold0.npy
    python eval.py -s config/dsensdp_ex.yaml -w path/to/model_run0_fold0.pt --shapley
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

print("Standard imports OK", flush=True)

try:
    from code.training.utils import output_name
    from code.datasets.utils import create_dataloader
    from utils.metrics import compute_metrics, get_random_baseline_metrics
    from utils.checkpoint import build_model, load_test_data, resolve_weights_path
    from shap_analysis.shapley import run_subset_inference, compute_shapley_per_metric
    print("Project imports OK", flush=True)
except Exception:
    print(f"\nImport error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


# ==============================================================================
#  Helpers
# ==============================================================================

def log(msg: str):
    print(msg, flush=True)


def _print_metrics(metrics: dict, title: str = "Metrics"):
    sep = "-" * 48
    log(f"\n{sep}")
    log(f"  {title}")
    log(sep)
    for k, v in metrics.items():
        log(f"  {k:<24}: {v:.4f}")
    log(sep)


def _save_results_csv(
    out_dir: Path,
    run_id: int,
    fold_id: int,
    metrics: dict,
    shapley_per_metric: dict = None,
):
    """
    Save metrics and optionally Shapley values to CSV files.

    Args:
        shapley_per_metric: {metric_name: {view_name: shapley_value}}
                            as returned by compute_shapley_per_metric.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_run{run_id}_fold{fold_id}.csv"
    pd.DataFrame([{"run_id": run_id, "fold_id": fold_id, **metrics}]).to_csv(metrics_path, index=False)
    log(f"\nMetrics saved -> {metrics_path}")

    if shapley_per_metric:
        rows = [
            {"run_id": run_id, "fold_id": fold_id,
             "metric": metric_name, "view": view, "shapley_value": phi}
            for metric_name, sv in shapley_per_metric.items()
            for view, phi in sv.items()
        ]
        shapley_path = out_dir / f"shapley_run{run_id}_fold{fold_id}.csv"
        pd.DataFrame(rows).to_csv(shapley_path, index=False)
        log(f"Shapley values saved -> {shapley_path}")


# ==============================================================================
#  Main inference pipeline
# ==============================================================================

def run_inference(args):

    log(f"\nReading config: {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    run_id  = args.run_id
    fold_id = args.fold_id

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if args.weights_file:
        weights_path = Path(args.weights_file)
    else:
        weights_path = resolve_weights_path(config, run_id, fold_id)
        log(f"  Reconstructed checkpoint path: {weights_path}")

    if not weights_path.exists():
        log(f"\nERROR: checkpoint not found: {weights_path}")
        log("  Check --weights_file or ensure save_weights=True was set during training.")
        sys.exit(1)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log(f"\nLoading checkpoint: {weights_path}")
    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)
    log(
        f"  run_id     = {checkpoint.get('run_id',    run_id)}\n"
        f"  fold_id    = {checkpoint.get('fold_id',   fold_id)}\n"
        f"  epoch      = {checkpoint.get('epoch',     '?')}\n"
        f"  best_score = {checkpoint.get('best_score','?')}"
    )

    # ── Reconstruct model & test set ──────────────────────────────────────────
    log("\nRebuilding model...")
    method = build_model(checkpoint, ckpt_config)

    log("\nLoading test set...")
    data_te = load_test_data(checkpoint, ckpt_config)

    BS        = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type = ckpt_config.get("task_type", "")
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    # ── Full-set inference ────────────────────────────────────────────────────
    log(f"\nRunning inference (batch_size={BS})...")
    t0 = time.time()
    with torch.no_grad():
        outputs = method.transform(
            create_dataloader(data_te, batch_size=BS, train=False),
            out_norm=output_name(task_type),
            not_return_repre=True,
        )
    log(f"  Done in {time.time() - t0:.2f}s")

    y_true  = data_te.get_all_labels()
    y_pred  = outputs["prediction"]
    metrics = compute_metrics(y_true, y_pred, task_type)
    _print_metrics(metrics, title=f"Test - run {run_id} / fold {fold_id}")

    # ── Shapley values (optional) ─────────────────────────────────────────────
    shapley_per_metric = None
    if args.shapley:
        view_names  = ckpt_config["experiment"]["preprocess"]["view_names"]
        metric_keys = list(metrics.keys())
        log(f"\nComputing Shapley values over {len(view_names)} views: {view_names}")

        baseline = get_random_baseline_metrics(y_true, task_type)
        v_dict   = run_subset_inference(
            method, data_te, y_true, view_names, task_type, BS,
            baseline, metrics, verbose=True,
        )
        shapley_per_metric = compute_shapley_per_metric(v_dict, view_names, metric_keys)

        sep = "-" * 48
        for metric_name, sv in shapley_per_metric.items():
            log(f"\n  Shapley ({metric_name}):")
            log(sep)
            for view, phi in sorted(sv.items(), key=lambda x: -abs(x[1])):
                sign = "+" if phi >= 0 else ""
                log(f"    {view:<20} {sign}{phi:.4f}")
            log(sep)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.save_predictions).parent if args.save_predictions else Path("preds")

    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, y_pred)
        log(f"\nPredictions (.npy) -> {pred_path}")

    _save_results_csv(out_dir, run_id, fold_id, metrics, shapley_per_metric)

    log("\nDone.")
    return metrics


# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Inference and evaluation of a MultiFusion checkpoint.")
    p.add_argument("--settings_file", "-s", required=True,
                   help="YAML config file (same as used for training).")
    p.add_argument("--weights_file",  "-w", default=None,
                   help="Explicit path to the .pt checkpoint (optional).")
    p.add_argument("--run_id",        "-r", type=int, default=0)
    p.add_argument("--fold_id",       "-f", type=int, default=0)
    p.add_argument("--batch_size",    "-b", type=int, default=None,
                   help="Inference batch size (default: value from config).")
    p.add_argument("--shapley",       action="store_true",
                   help="Compute and save per-view Shapley values.")
    p.add_argument("--save_predictions", default=None, metavar="PATH",
                   help="Path to save raw predictions (.npy). "
                        "The parent directory also receives the results CSV.")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_inference(args)
    except SystemExit:
        raise
    except Exception:
        log("\nUnhandled error:\n")
        log(traceback.format_exc())
        sys.exit(1)