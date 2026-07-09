"""
compute_subset_metrics.py
--------------------------
Compute F1-weighted and accuracy for all 2^K - 1 subsets of modalities,
averaged over K-fold cross-validation.

Usage:
    python -m scripts.vdict.subset_metrics -s config/dsensdp_ex.yaml \
        --fold_ids 0 1 2 3 4 --out preds/eval/subset_metrics.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--settings", required=True)
    p.add_argument("--fold_ids", nargs="+", type=int, default=[0])
    p.add_argument("--run_id", "-r", type=int, default=0)
    p.add_argument("--out", default="preds/eval/subset_metrics.csv")
    return p.parse_args()


def main(args):
    with open(args.settings) as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils.checkpoint import build_model, load_test_data, resolve_weights_path
    from utils.metrics import compute_metrics

    view_names = config["experiment"]["preprocess"]["view_names"]
    n_views    = len(view_names)

    # All non-empty subsets
    all_subsets = []
    for size in range(1, n_views + 1):
        for combo in itertools.combinations(view_names, size):
            all_subsets.append(list(combo))

    print(f"Computing metrics for {len(all_subsets)} subsets × "
          f"{len(args.fold_ids)} fold(s)...\n")

    # Accumulate metrics per subset per fold
    results = {frozenset(s): {"f1_weighted": [], "accuracy": []}
               for s in all_subsets}

    from code.datasets.utils import create_dataloader
    from code.training.utils import output_name

    task_type = config.get("task_type", "classification")

    for fold_id in args.fold_ids:
        print(f"── Fold {fold_id} ──────────────────────────────")
        weights_path = resolve_weights_path(config, args.run_id, fold_id)
        print(f"  Checkpoint: {weights_path}", flush=True)

        import torch
        checkpoint = torch.load(weights_path, map_location="cpu")

        data_te = load_test_data(checkpoint, config)
        labels  = data_te.get_all_labels()
        method  = build_model(checkpoint, config)

        loader = create_dataloader(data_te, batch_size=256, train=False)

        for subset_list in all_subsets:
            label    = "+".join(subset_list)
            args_fwd = {
                "inference_views": subset_list,
                "missing_method":  method.missing_method,
            }
            out = method.transform(
                loader,
                out_norm=output_name(task_type),
                args_forward=args_fwd,
                perc_forward=1.0,
                not_return_repre=True,
            )
            preds   = out["prediction"]
            metrics = compute_metrics(labels, preds, task_type)
            results[frozenset(subset_list)]["f1_weighted"].append(
                metrics["f1_weighted"])
            results[frozenset(subset_list)]["accuracy"].append(
                metrics["accuracy"])
            print(f"  {label:<45}  "
                  f"f1_w={metrics['f1_weighted']:.4f}  "
                  f"acc={metrics['accuracy']:.4f}")

    # Build summary dataframe
    rows = []
    for subset_list in all_subsets:
        key  = frozenset(subset_list)
        f1s  = results[key]["f1_weighted"]
        accs = results[key]["accuracy"]
        rows.append({
            "subset":        "+".join(subset_list),
            "n_modalities":  len(subset_list),
            "modalities":    ", ".join(subset_list),
            "f1_weighted_mean": np.mean(f1s),
            "f1_weighted_std":  np.std(f1s),
            "accuracy_mean":    np.mean(accs),
            "accuracy_std":     np.std(accs),
        })

    df = pd.DataFrame(rows).sort_values(
        ["n_modalities", "f1_weighted_mean"], ascending=[True, False])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")

    # Pretty print
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Subset performance (mean ± std over {len(args.fold_ids)} fold(s))")
    print(sep)
    for n in range(1, n_views + 1):
        sub = df[df["n_modalities"] == n]
        print(f"\n  ── {n} modalité(s) ──")
        for _, row in sub.iterrows():
            print(f"    {row['subset']:<45}  "
                  f"F1w={row['f1_weighted_mean']:.4f}±{row['f1_weighted_std']:.4f}  "
                  f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}")
    print(f"\n{sep}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main(parse_args())