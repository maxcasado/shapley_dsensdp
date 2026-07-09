"""
run_perceptual_score.py
------------------------
Compute the permutation-based Perceptual Score (Gat et al., NeurIPS 2021)
for a trained checkpoint, per fold.

Usage:
    python -m scripts.attribution.perceptual_score -s config/com_average.yaml --fold_ids 0 1 2 3 4 \
        --out_dir results/com --n_perm 10

    python -m scripts.attribution.perceptual_score -s config/com_geo.yaml --fold_ids 0 1 2 3 4 \
        --fixed_views geo --out_dir results/com_geo --n_perm 10
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

try:
    from code.training.utils import output_name
    from code.datasets.utils import create_dataloader
    from utils.metrics import compute_metrics
    from utils.checkpoint import build_model, load_test_data, resolve_weights_path
    from shap_analysis.perceptual_score import compute_perceptual_score_fold
except Exception:
    print(f"\nImport error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


def log(msg: str):
    print(msg, flush=True)


def _run_fold(run_id: int, fold_id: int, args, config: dict):
    log(f"\n{'─'*52}\n  Run {run_id} / Fold {fold_id} — Perceptual Score\n{'─'*52}")

    weights_path = resolve_weights_path(config, run_id, fold_id)
    log(f"  Checkpoint: {weights_path}")
    if not weights_path.exists():
        log("  ERROR: checkpoint not found, skipping.")
        return None

    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)

    method  = build_model(checkpoint, ckpt_config)
    data_te = load_test_data(checkpoint, ckpt_config)

    BS        = args.batch_size or ckpt_config["training"]["batch_size"]
    task_type = ckpt_config.get("task_type", "")
    method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    view_names  = ckpt_config["experiment"]["preprocess"]["view_names"]
    fixed_views = args.fixed_views or []
    permute_modalities = [v for v in view_names if v not in fixed_views]

    t0 = time.time()
    with torch.no_grad():
        out_full = method.transform(
            create_dataloader(data_te, batch_size=BS, train=False),
            out_norm=output_name(task_type),
            not_return_repre=True,
        )
    y_true       = data_te.get_all_labels()
    full_metrics = compute_metrics(y_true, out_full["prediction"], task_type)
    p_pos        = float(np.mean(y_true == 1))
    acc_majority = max(p_pos, 1 - p_pos)
    log(f"  Full-coalition inference: {time.time()-t0:.2f}s  "
        f"{args.metric}(all)={full_metrics[args.metric]:.4f}  "
        f"accuracy_majority_vote={acc_majority:.4f}")

    t1 = time.time()
    log(f"  Permuting {len(permute_modalities)} modalities x {args.n_perm} perms "
        f"(fixed: {fixed_views or 'none'})")
    rows = compute_perceptual_score_fold(
        method, data_te, y_true, view_names, permute_modalities, task_type, BS,
        f1_all=full_metrics[args.metric], n_perm=args.n_perm,
        seed=args.seed + fold_id, metric_name=args.metric,
    )
    log(f"  Perceptual Score computed in {time.time()-t1:.2f}s")

    for r in rows:
        r["PS_task"]  = r["PS_raw"] / (1 - acc_majority) if (1 - acc_majority) > 0 else float("nan")
        r["PS_model"] = r["PS_raw"] / r["f1_all"] if r["f1_all"] != 0 else float("nan")
        r["fold_id"]  = fold_id
        r["run_id"]   = run_id
        r["metric"]   = args.metric
        r["accuracy_majority_vote"] = acc_majority

    return rows


def run(args):
    log(f"\nReading config: {args.settings_file}")
    with open(args.settings_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_total = time.time()
    for fold_id in args.fold_ids:
        out_csv = out_dir / f"perceptual_score_fold{fold_id}.csv"
        if out_csv.exists() and not args.overwrite:
            log(f"\nFold {fold_id}: already computed -> {out_csv} (loading, use --overwrite to redo).")
            all_rows.append(pd.read_csv(out_csv))
            continue

        rows = _run_fold(args.run_id, fold_id, args, config)
        if rows is None:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        log(f"  Saved -> {out_csv}")
        all_rows.append(df)

    if not all_rows:
        log("\nNo folds completed successfully.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    summary = (
        df_all.groupby("modality")[["PS_raw", "PS_task", "PS_model"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["n_folds"] = df_all.groupby("modality")["fold_id"].nunique().values
    summary_path = out_dir / "perceptual_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"\nSummary saved -> {summary_path}")
    log(f"Total wall-clock time: {time.time() - t_total:.2f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Compute permutation-based Perceptual Score.")
    p.add_argument("--settings_file", "-s", required=True)
    p.add_argument("--run_id", "-r", type=int, default=0)
    p.add_argument("--fold_ids", type=int, nargs="+", default=[0])
    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--out_dir", "-o", default="results/ps")
    p.add_argument("--fixed_views", nargs="+", default=None,
                   help="Views never permuted (e.g. geo — fixed infrastructure).")
    p.add_argument("--n_perm", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metric", default="f1_weighted",
                   help="Metric key used as 'F1' for PS_raw/PS_task/PS_model.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except SystemExit:
        raise
    except Exception:
        log("\nUnhandled error:\n")
        log(traceback.format_exc())
        sys.exit(1)
