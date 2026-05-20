"""
eval.py
-------
Load saved checkpoints and evaluate metrics on the associated test sets.
Optionally computes Shapley values per view.

Usage:
    # Single fold
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0
    python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0 --shapley

    # All folds — mean +- std reported
    python eval.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4
    python eval.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 --shapley
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
    from shap_analysis.shapley import (run_subset_inference,
                                       compute_shapley_per_metric,
                                       compute_shapley_interactions_metrics)
    print("Project imports OK", flush=True)
except Exception:
    print(f"\nImport error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


def log(msg: str):
    print(msg, flush=True)


# ==============================================================================
#  Helpers
# ==============================================================================

def _print_metrics(metrics: dict, title: str = "Metrics"):
    sep = "-" * 48
    log(f"\n{sep}\n  {title}\n{sep}")
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
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"run_id": run_id, "fold_id": fold_id, **metrics}]).to_csv(
        out_dir / f"metrics_run{run_id}_fold{fold_id}.csv", index=False
    )

    if shapley_per_metric:
        rows = [
            {"run_id": run_id, "fold_id": fold_id,
             "metric": metric_name, "view": view, "shapley_value": phi}
            for metric_name, sv in shapley_per_metric.items()
            for view, phi in sv.items()
        ]
        pd.DataFrame(rows).to_csv(
            out_dir / f"shapley_run{run_id}_fold{fold_id}.csv", index=False
        )


def _print_aggregated(all_metrics: list, all_shapley: list, all_interactions: list):
    """Print mean ± std across folds for metrics, Shapley values, and SII."""
    sep = "=" * 52

    # ── Metrics ───────────────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics)
    log(f"\n{sep}\n  Metrics — mean ± std across {len(all_metrics)} fold(s)\n{sep}")
    for col in df_metrics.columns:
        if col in ("run_id", "fold_id"):
            continue
        log(f"  {col:<24}: {df_metrics[col].mean():.4f} ± {df_metrics[col].std():.4f}")
    log(sep)

    # ── Shapley values ────────────────────────────────────────────────────────
    if all_shapley:
        df_shap = pd.concat([pd.DataFrame(s) for s in all_shapley], ignore_index=True)
        log(f"\n{sep}\n  Shapley values — mean ± std across {len(all_shapley)} fold(s)\n{sep}")
        for metric_name, grp in df_shap.groupby("metric"):
            log(f"\n  [{metric_name}]")
            view_stats = grp.groupby("view")["shapley_value"].agg(["mean", "std"])
            view_stats = view_stats.sort_values("mean", ascending=False)
            for view, row in view_stats.iterrows():
                sign = "+" if row["mean"] >= 0 else ""
                log(f"    {view:<20} {sign}{row['mean']:.4f} ± {row['std']:.4f}")
        log(sep)

    # ── SII ───────────────────────────────────────────────────────────────────
    if all_interactions:
        df_sii = pd.concat([pd.DataFrame(s) for s in all_interactions], ignore_index=True)
        log(f"\n{sep}\n  Shapley Interaction Index (SII) — mean ± std across {len(all_interactions)} fold(s)\n{sep}")
        for metric_name, grp in df_sii.groupby("metric"):
            log(f"\n  [{metric_name}]")
            pair_stats = grp.groupby(["view_i", "view_j"])["sii_value"].agg(["mean", "std"])
            pair_stats = pair_stats.sort_values("mean", key=abs, ascending=False)
            for (vi, vj), row in pair_stats.iterrows():
                sign = "+" if row["mean"] >= 0 else ""
                log(f"    {vi:<12} × {vj:<12}  {sign}{row['mean']:.4f} ± {row['std']:.4f}")
        log(sep)

    return df_metrics


def _save_aggregated_csv(out_dir: Path, run_id: int,
                         all_metrics: list,
                         all_shapley: list,
                         all_interactions: list):
    """Save mean ± std summary CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics summary
    df_metrics = pd.DataFrame(all_metrics)
    summary_rows = []
    for col in df_metrics.columns:
        if col in ("run_id", "fold_id"):
            continue
        summary_rows.append({
            "metric":  col,
            "mean":    df_metrics[col].mean(),
            "std":     df_metrics[col].std(),
            "n_folds": len(df_metrics),
        })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"metrics_summary_run{run_id}.csv", index=False)
    log(f"Metrics summary saved     -> {out_dir / f'metrics_summary_run{run_id}.csv'}")

    # Shapley values summary
    if all_shapley:
        df_shap = pd.concat([pd.DataFrame(s) for s in all_shapley], ignore_index=True)
        shap_summary = (
            df_shap.groupby(["metric", "view"])["shapley_value"]
            .agg(mean="mean", std="std", n_folds="count")
            .reset_index()
        )
        shap_summary["run_id"] = run_id
        shap_summary.to_csv(
            out_dir / f"shapley_summary_run{run_id}.csv", index=False)
        log(f"Shapley summary saved     -> {out_dir / f'shapley_summary_run{run_id}.csv'}")

    # SII summary
    if all_interactions:
        df_sii = pd.concat([pd.DataFrame(s) for s in all_interactions], ignore_index=True)
        sii_summary = (
            df_sii.groupby(["metric", "view_i", "view_j"])["sii_value"]
            .agg(mean="mean", std="std", n_folds="count")
            .reset_index()
        )
        sii_summary["run_id"] = run_id
        sii_summary.to_csv(
            out_dir / f"sii_summary_run{run_id}.csv", index=False)
        log(f"SII summary saved         -> {out_dir / f'sii_summary_run{run_id}.csv'}")


# ==============================================================================
#  Per-fold computation
# ==============================================================================

def _run_fold(run_id: int, fold_id: int, args, config: dict) -> tuple:
    """
    Run inference + optional Shapley for one fold.

    Returns:
        metrics_row        : dict  {metric_name: value, run_id, fold_id}
        shapley_rows       : list of dicts (empty if --shapley not set)
        shapley_per_metric : dict {metric: {view: phi}} or None
    """
    log(f"\n{'─'*52}")
    log(f"  Run {run_id} / Fold {fold_id}")
    log(f"{'─'*52}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args.weights_file and len(args.fold_ids) == 1:
        weights_path = Path(args.weights_file)
    else:
        weights_path = resolve_weights_path(config, run_id, fold_id)
    log(f"  Checkpoint: {weights_path}")

    if not weights_path.exists():
        log(f"  ERROR: checkpoint not found, skipping.")
        return None, None, None

    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)
    log(f"  epoch={checkpoint.get('epoch','?')}  "
        f"best_score={checkpoint.get('best_score','?'):.4f}")

    method  = build_model(checkpoint, ckpt_config)
    data_te = load_test_data(checkpoint, ckpt_config)

    BS        = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type = ckpt_config.get("task_type", "")
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        outputs = method.transform(
            create_dataloader(data_te, batch_size=BS, train=False),
            out_norm=output_name(task_type),
            not_return_repre=True,
        )
    log(f"  Inference done in {time.time() - t0:.2f}s")

    y_true  = data_te.get_all_labels()
    metrics = compute_metrics(y_true, outputs["prediction"], task_type)
    _print_metrics(metrics, title=f"Fold {fold_id}")

    metrics_row = {"run_id": run_id, "fold_id": fold_id, **metrics}

    # ── Shapley values and/or SII (optional) ─────────────────────────────────
    shapley_rows     = []
    interaction_rows = []

    if args.shapley or args.interactions:
        view_names  = ckpt_config["experiment"]["preprocess"]["view_names"]
        metric_keys = list(metrics.keys())
        log(f"  Computing Shapley over {len(view_names)} views: {view_names}")

        baseline = get_random_baseline_metrics(y_true, task_type)
        v_dict   = run_subset_inference(
            method, data_te, y_true, view_names, task_type, BS,
            baseline, metrics, verbose=True,
        )

        # Shapley values
        if args.shapley:
            sv_per_metric = compute_shapley_per_metric(v_dict, view_names, metric_keys)
            sep = "-" * 48
            for metric_name, sv in sv_per_metric.items():
                log(f"\n  Shapley values [{metric_name}]:\n{sep}")
                for view, phi in sorted(sv.items(), key=lambda x: -x[1]):
                    sign = "+" if phi >= 0 else ""
                    log(f"    {view:<20} {sign}{phi:.4f}")
                log(sep)
                for view, phi in sv.items():
                    shapley_rows.append({
                        "run_id": run_id, "fold_id": fold_id,
                        "metric": metric_name, "view": view, "shapley_value": phi,
                    })

        # SII
        if args.interactions:
            sii = compute_shapley_interactions_metrics(v_dict, view_names, metric_keys)
            sep = "-" * 52
            for metric_name, pairs in sii.items():
                log(f"\n  SII [{metric_name}]:\n{sep}")
                for (vi, vj), val in sorted(pairs.items(), key=lambda x: -abs(x[1])):
                    sign = "+" if val >= 0 else ""
                    log(f"    {vi:<12} × {vj:<12}  {sign}{val:.4f}")
                log(sep)
                for (vi, vj), val in pairs.items():
                    interaction_rows.append({
                        "run_id": run_id, "fold_id": fold_id,
                        "metric": metric_name,
                        "view_i": vi, "view_j": vj, "sii_value": val,
                    })

    return metrics_row, shapley_rows, interaction_rows


# ==============================================================================
#  Entry point
# ==============================================================================

def run_inference(args):
    log(f"\nReading config: {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    out_dir  = Path(args.out_dir)
    run_id   = args.run_id
    fold_ids = args.fold_ids

    all_metrics      = []
    all_shapley      = []
    all_interactions = []

    for fold_id in fold_ids:
        metrics_row, shapley_rows, interaction_rows = _run_fold(
            run_id, fold_id, args, config
        )
        if metrics_row is None:
            continue

        all_metrics.append(metrics_row)
        if shapley_rows:
            all_shapley.append(shapley_rows)
        if interaction_rows:
            all_interactions.append(interaction_rows)

        # Save per-fold CSVs
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{k: v for k, v in metrics_row.items()
                       if k not in ("run_id", "fold_id")}]).to_csv(
            out_dir / f"metrics_run{run_id}_fold{fold_id}.csv", index=False)
        if shapley_rows:
            pd.DataFrame(shapley_rows).to_csv(
                out_dir / f"shapley_run{run_id}_fold{fold_id}.csv", index=False)
        if interaction_rows:
            pd.DataFrame(interaction_rows).to_csv(
                out_dir / f"sii_run{run_id}_fold{fold_id}.csv", index=False)

    if not all_metrics:
        log("\nNo folds completed successfully.")
        return None

    if len(fold_ids) > 1:
        _print_aggregated(all_metrics, all_shapley, all_interactions)
        _save_aggregated_csv(out_dir, run_id, all_metrics, all_shapley, all_interactions)

    log("\nDone.")
    return all_metrics


# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MultiFusion checkpoints on their test sets."
    )
    p.add_argument("--settings_file", "-s", required=True)
    p.add_argument("--weights_file",  "-w", default=None,
                   help="Explicit .pt path (single-fold only).")
    p.add_argument("--run_id", "-r", type=int, default=0)

    fold_group = p.add_mutually_exclusive_group()
    fold_group.add_argument("--fold_id",  "-f", type=int, default=None,
                            help="Single fold to evaluate.")
    fold_group.add_argument("--fold_ids", type=int, nargs="+", default=None,
                            metavar="FOLD",
                            help="Multiple folds (e.g. --fold_ids 0 1 2 3 4).")

    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--out_dir", "-o", default="preds/eval",
                   help="Output directory for result CSVs (default: preds/eval).")
    p.add_argument("--shapley", action="store_true",
                   help="Compute per-view Shapley values.")
    p.add_argument("--interactions", action="store_true",
                   help="Compute Shapley Interaction Index (SII) per pair of views.")
    p.add_argument("--save_predictions", default=None, metavar="PATH",
                   help="Path to save raw predictions (.npy).")
    args = p.parse_args()

    if args.fold_ids is not None:
        pass
    elif args.fold_id is not None:
        args.fold_ids = [args.fold_id]
    else:
        args.fold_ids = [0]

    return args


if __name__ == "__main__":
    try:
        run_inference(parse_args())
    except SystemExit:
        raise
    except Exception:
        log("\nUnhandled error:\n")
        log(traceback.format_exc())
        sys.exit(1)