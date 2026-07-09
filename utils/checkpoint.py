"""
utils/checkpoint.py
-------------------
Shared utilities for loading a saved checkpoint and reconstructing
the model and test set from it.

Imported by the checkpoint-driven entry points: scripts/vdict/build_vdicts.py
(v-dict generation) and scripts/spatial/explain.py (spatial Shapley inference).
"""

import copy
from pathlib import Path

import numpy as np
import torch

from code.datasets.utils import load_structure
from code.training.utils import assign_labels_weights


def build_model(checkpoint: dict, config: dict):
    """
    Reconstruct a model from a saved checkpoint without running any training.

    Loads the dataset to infer the architecture, then restores the state_dict.

    Args:
        checkpoint: Dict loaded from a .pt file (output of torch.load).
        config:     Experiment config dict (typically checkpoint["config"]).

    Returns:
        model set to eval() mode with weights restored.
    """
    from code.training.learn_pipeline import build_model as _build

    cfg       = copy.deepcopy(config)
    input_dir = cfg["input_dir_folder"]
    data_name = cfg["data_name"]

    print(f"  Loading data to build architecture ({data_name})...", flush=True)
    data_views = load_structure(input_dir, data_name, load_memory=cfg.get("load_memory", False), views_used=cfg.get("experiment", {}).get("preprocess", {}).get("view_names", []))
    data_views.load_stats(input_dir, data_name)
    data_views.set_additional_info(**cfg["experiment"].get("preprocess", {}))

    if cfg.get("task_type", "").lower() in ["classification", "multilabel"]:
        assign_labels_weights(cfg, data_views)

    print("  Building model (no training)...", flush=True)
    method = _build(data_views, **cfg)

    print("  Loading state_dict...", flush=True)
    method.load_state_dict(checkpoint["model_state_dict"])
    method.eval()
    print("  State-dict loaded.", flush=True)
    return method


def load_test_data(checkpoint: dict, config: dict):
    """
    Load the test set associated with a checkpoint.

    For fixed train/test splits (data_name contains "train"):
        loads the corresponding test split directly.

    For k-fold splits:
        loads the full dataset and applies the fold mask stored in the
        checkpoint's test_indices_file.

    Args:
        checkpoint: Dict loaded from a .pt file.
        config:     Experiment config dict.

    Returns:
        Dataset_MultiView filtered to the test samples of this fold.

    Raises:
        FileNotFoundError: if the test indices file is missing for a k-fold run.
    """
    input_dir = config["input_dir_folder"]
    data_name = config["data_name"]

    if "train" in data_name:
        test_name = data_name.replace("train", "test")
        print(f"  Fixed split -> loading: {test_name}", flush=True)
        data_te = load_structure(input_dir, test_name, load_memory=config.get("load_memory", False), views_used=config.get("experiment", {}).get("preprocess", {}).get("view_names", []))
        data_te.load_stats(input_dir, data_name)
    else:
        indices_file = checkpoint.get("test_indices_file")
        if not indices_file or not Path(indices_file).exists():
            raise FileNotFoundError(
                f"Test indices not found ({indices_file}). "
                "Re-run training with save_test_set=True."
            )
        print(f"  K-fold -> indices from: {indices_file}", flush=True)
        test_indices = np.load(indices_file, allow_pickle=True)
        data_te = load_structure(input_dir, data_name, load_memory=config.get("load_memory", False), views_used=config.get("experiment", {}).get("preprocess", {}).get("view_names", []))
        data_te.load_stats(input_dir, data_name)
        data_te.set_val_mask(test_indices)
        data_te.set_data_mode(train=False)

    data_te.set_additional_info(**config["experiment"].get("preprocess", {}))
    print(f"  {len(data_te)} test samples.", flush=True)
    return data_te


def resolve_weights_path(config: dict, run_id: int, fold_id: int) -> Path:
    """
    Reconstruct the checkpoint path from the config and run/fold ids.

    Args:
        config:  Experiment config dict.
        run_id:  Run index.
        fold_id: Fold index.

    Returns:
        Path to the .pt checkpoint file.
    """
    from code.training.utils import assign_multifusion_name

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
    return Path(weights_dir) / method_name / f"model_run{run_id}_fold{fold_id}.pt"