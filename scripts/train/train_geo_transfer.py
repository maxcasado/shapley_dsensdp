"""
train_geo_transfer.py
---------------------
Train a model on one geographic region and evaluate on another.
No k-fold — uses identifier-based train/test splits.

Usage:
    python -m scripts.train.train_geo_transfer -s config/com_average_multi.yaml \
        --train_region region_3_identifiers.npy \
        --test_region  region_0_identifiers.npy \
        --out_suffix   R3_to_R0

    python -m scripts.train.train_geo_transfer -s config/com_average_multi.yaml \
        --train_region region_1_identifiers.npy \
        --test_region  region_0_identifiers.npy \
        --out_suffix   R1_to_R0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from code.training.learn_pipeline import MultiFusion_train
from code.training.utils import (assign_multifusion_name, output_name,
                                  assign_labels_weights)
from code.datasets.views_structure import Dataset_MultiView
from code.datasets.utils import create_dataloader, load_structure
from utils.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--settings_file", "-s", required=True)
    p.add_argument("--train_region",  required=True,
                   help="Path to .npy file with train identifiers")
    p.add_argument("--test_region",   required=True,
                   help="Path to .npy file with test identifiers")
    p.add_argument("--out_suffix",    default="geo_transfer",
                   help="Suffix appended to output directory name")
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    with open(args.settings_file) as f:
        config = yaml.safe_load(f)

    # ── Load region identifiers ───────────────────────────────────────────────
    train_ids = set(np.load(args.train_region, allow_pickle=True).tolist())
    test_ids  = set(np.load(args.test_region,  allow_pickle=True).tolist())
    print(f"Train region : {len(train_ids)} points")
    print(f"Test  region : {len(test_ids)}  points")

    # ── Load full dataset ─────────────────────────────────────────────────────
    input_dir  = config["input_dir_folder"]
    output_dir = config["output_dir_folder"].rstrip("/") + f"_geo/{args.out_suffix}"
    data_name  = config["data_name"]

    data_views = load_structure(
        input_dir, data_name,
        load_memory=config.get("load_memory", False),
        views_used=config.get("experiment", {})
                         .get("preprocess", {})
                         .get("view_names", []),
    )
    data_views.load_stats(input_dir, data_name)

    all_ids = np.array(data_views.get_all_identifiers())
    print(f"Full dataset : {len(all_ids)} points")

    # ── Build train / test index arrays ──────────────────────────────────────
    train_idx = np.array([i for i, id_ in enumerate(all_ids) if id_ in train_ids])
    test_idx  = np.array([i for i, id_ in enumerate(all_ids) if id_ in test_ids])
    print(f"Matched train: {len(train_idx)}  test: {len(test_idx)}")

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("No matching identifiers found — check .npy files and dataset.")

    # ── Prepare train/test splits ─────────────────────────────────────────────
    import copy
    data_views.set_val_mask(all_ids[test_idx])   # set test mask by identifiers
    data_te = copy.deepcopy(data_views)
    data_te.set_data_mode(train=False)
    data_va = data_te  # use test as validation for early stopping

    data_views.set_additional_info(**config["experiment"].get("preprocess"))
    data_te.set_additional_info(**config["experiment"].get("preprocess"))

    print(f"Training on {len(data_views)} samples, testing on {len(data_te)} samples")

    # ── Loss / weights ────────────────────────────────────────────────────────
    if "loss_args" not in config["training"]:
        config["training"]["loss_args"] = {}
    config["training"]["loss_args"].setdefault(
        "name", "ce" if config.get("task_type", "").lower() == "classification" else "bce"
    )
    if config.get("task_type", "").lower() in ["classification", "multilabel"]:
        assign_labels_weights(config, data_views)

    BS = config["training"]["batch_size"]
    r_seed = config["experiment"].get("runs_seed", [42])[0]
    np.random.seed(r_seed)

    method_name = assign_multifusion_name(
        config["training"], config["method"],
        more_info_str=config.get("additional_method_name", "") + f"-{args.out_suffix}",
    )
    print(f"METHOD_NAME = '{method_name}'")

    # ── Save test indices ─────────────────────────────────────────────────────
    testset_dir = Path(config.get("testset_dir", f"{output_dir}/test_sets/{data_name}")) / method_name
    testset_dir.mkdir(parents=True, exist_ok=True)
    np.save(testset_dir / "test_indices_run0_fold0.npy", all_ids[test_idx])
    np.save(testset_dir / "test_labels_run0_fold0.npy",  data_te.get_all_labels())
    np.save(testset_dir / "test_identifiers_run0_fold0.npy", data_te.get_all_identifiers())

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    method, trainer = MultiFusion_train(
        data_views, val_data=data_va, run_id=0, fold_id=0,
        method_name=method_name, **config,
    )
    training_time = time.time() - t0
    print(f"Training done in {training_time:.1f}s")

    # ── Save weights ──────────────────────────────────────────────────────────
    if config.get("save_weights", False):
        weights_dir = Path(config.get("weights_dir", f"{output_dir}/weights")) / method_name
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "run_id": 0, "fold_id": 0,
            "model_state_dict": method.state_dict(),
            "config": config,
            "seed": r_seed,
            "train_region": str(args.train_region),
            "test_region":  str(args.test_region),
            "test_indices_file": str(testset_dir / "test_indices_run0_fold0.npy"),
        }, weights_dir / "model_run0_fold0.pt")
        print(f"Weights saved -> {weights_dir}/model_run0_fold0.pt")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    outputs_te = method.transform(
        create_dataloader(data_te, batch_size=BS, train=False),
        out_norm=output_name(config.get("task_type", "classification")),
        not_return_repre=True,
    )
    y_true  = data_te.get_all_labels()
    metrics = compute_metrics(y_true, outputs_te["prediction"],
                              config.get("task_type"))

    print(f"\n{'='*52}")
    print(f"  Transfer: {args.train_region} → {args.test_region}")
    print(f"{'='*52}")
    for k, v in metrics.items():
        print(f"  {k:<22} {v:.4f}")
    print(f"{'='*52}")
    print(f"  Epochs:        {trainer.callbacks[0].stopped_epoch}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Total time:    {time.time()-start_time:.1f}s")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_dir = Path(f"{output_dir}/metadata/{data_name}/{method_name}")
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "out_suffix": args.out_suffix,
        "train_region": args.train_region,
        "test_region":  args.test_region,
        "n_train": len(train_idx),
        "n_test":  len(test_idx),
        "epoch_runs": trainer.callbacks[0].stopped_epoch,
        "training_time": training_time,
        **metrics,
    }]).to_csv(meta_dir / "metadata_transfer.csv", index=False)
    print(f"Metadata saved -> {meta_dir}/metadata_transfer.csv")


if __name__ == "__main__":
    main()