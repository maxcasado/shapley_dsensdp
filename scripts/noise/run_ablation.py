"""
run_noise_ablation.py
----------------------
Self-contained Gaussian-noise ablation on the S2_S2VI modality of the CoM
model. For sigma in {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0} and each of the 5
folds it trains the model (config/com_average.yaml), computes exact Shapley
values (16 coalitions) and the permutation Perceptual Score (10 perms), then
aggregates phi/PS vs sigma into a summary CSV and a two-panel figure.

This file does not import noise_injection.py or any earlier
run_noise_ablation.py. It only imports the existing, unmodified
training/eval machinery (code/datasets, code/training, shap_analysis,
utils) -- the same machinery train_multi.py and run_perceptual_score.py use.

Where noise is injected
------------------------
z-score normalization happens in Dataset_MultiView.normalize_w_stats
(code/datasets/views_structure.py), called from Dataset_MultiView.__getitem__
(the per-sample read path used by the DataLoader). "S2_S2VI" is a composite
view: code/datasets/utils.py:xray_to_dataviews expands it into raw views
"S2" and "S2VI"; __getitem__ normalizes each independently and only
concatenates them into the "S2_S2VI" key right before returning the sample
(see the `views_to_add` / concatenate block at the end of __getitem__).

So __getitem__'s return value is exactly "after normalization, before the
tensor is returned by the dataloader" for S2_S2VI. We monkey-patch
Dataset_MultiView.__getitem__ at runtime (never touching the file on disk)
to add N(0, sigma^2) noise to the concatenated "S2_S2VI" array right before
it is returned. At sigma=0.0 the patch is a strict no-op (the unmodified
dict is returned untouched). The patch stays active for training and for
Shapley/PS evaluation alike, since both go through this same class.

Noise RNG independence
-----------------------
The fold split uses the legacy global `np.random` (np.random.seed +
np.random.shuffle), exactly as train_multi.py does. The noise patch never
touches global np.random state: for every sample it draws from a fresh
`np.random.default_rng(seed)` keyed deterministically off
(noise_seed_offset, fold_id, sigma, sample_index) -- independent of the
fold-split RNG and stable across DataLoader worker processes.
"""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.training.learn_pipeline import MultiFusion_train
from code.training.utils import assign_multifusion_name, output_name, assign_labels_weights
from code.datasets.views_structure import Dataset_MultiView
from code.datasets.utils import create_dataloader, load_structure
from utils.metrics import compute_metrics, get_random_baseline_metrics
from utils.checkpoint import build_model, load_test_data
from shap_analysis.shapley import run_subset_inference, compute_shapley_per_metric
from shap_analysis.perceptual_score import compute_perceptual_score_fold


# ── Experiment constants ──────────────────────────────────────────────────────
SETTINGS_FILE = "config/com_average.yaml"
DRY_RUN       = False
SIGMAS        = [0.0, 1.0] if DRY_RUN else [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
N_PERM        = 10
METRIC_NAME   = "f1_weighted"
RUN_ID        = 0
OUT_ROOT      = Path("results/noise_ablation")

NOISE_SEED_OFFSET = 999_983   # arbitrary, far from experiment.runs_seed (10)
PS_SEED_OFFSET     = 424_242  # arbitrary, distinct from NOISE_SEED_OFFSET

MODALITY_COLORS = {
    "S2_S2VI": "#4e79a7",
    "S1":      "#f28e2b",
    "weather": "#e15759",
    "DEM":     "#76b7b2",
}


# ── Noise injection: runtime monkey-patch of Dataset_MultiView.__getitem__ ───
_ORIG_GETITEM = Dataset_MultiView.__getitem__
_NOISE_CTX = {"sigma": 0.0, "seed_base": 0}


def _make_seed_base(sigma: float, fold_id: int) -> int:
    return NOISE_SEED_OFFSET + fold_id * 7919 + int(round(sigma * 10_000))


def set_noise_context(sigma: float, fold_id: int):
    _NOISE_CTX["sigma"] = float(sigma)
    _NOISE_CTX["seed_base"] = _make_seed_base(sigma, fold_id)


def _noisy_getitem(self, index):
    item = _ORIG_GETITEM(self, index)
    sigma = _NOISE_CTX["sigma"]
    if sigma > 0.0 and "S2_S2VI" in item["views"]:
        arr = item["views"]["S2_S2VI"]
        rng = np.random.default_rng(_NOISE_CTX["seed_base"] + index)
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(arr.dtype)
        item["views"]["S2_S2VI"] = arr + noise
    return item


Dataset_MultiView.__getitem__ = _noisy_getitem


# ── Config helpers ─────────────────────────────────────────────────────────────
def build_config(base_config: dict, sigma: float) -> dict:
    config = copy.deepcopy(base_config)
    sigma_dir = OUT_ROOT / f"sigma_{sigma}"
    config["output_dir_folder"] = str(sigma_dir)
    config["weights_dir"] = str(sigma_dir / "weights")
    config["testset_dir"] = str(sigma_dir / "test_sets")
    config["save_weights"] = True
    config["save_test_set"] = True
    if DRY_RUN:
        config["training"]["max_epochs"] = 2

    if "loss_args" not in config["training"]:
        config["training"]["loss_args"] = {}
    task = config.get("task_type", "").lower()
    if task == "classification":
        config["training"]["loss_args"].setdefault("name", "ce")
    elif task == "multilabel":
        config["training"]["loss_args"].setdefault("name", "bce")
    return config


# ── Per-fold train + shapley + PS ──────────────────────────────────────────────
def run_fold(config: dict, sigma: float, fold_id: int, data_views_tr, val_indices, method_name: str):
    sigma_dir = Path(config["output_dir_folder"])
    view_names = config["experiment"]["preprocess"]["view_names"]
    task_type = config.get("task_type")
    BS = config["training"]["batch_size"]

    data_views_tr.set_val_mask(val_indices)
    data_views_te = copy.deepcopy(data_views_tr)
    data_views_te.set_data_mode(train=False)
    data_views_va = data_views_te

    data_views_tr.set_additional_info(**config["experiment"]["preprocess"])
    data_views_va.set_additional_info(**config["experiment"]["preprocess"])
    data_views_te.set_additional_info(**config["experiment"]["preprocess"])

    if task_type.lower() in ["classification", "multilabel"]:
        assign_labels_weights(config, data_views_tr)

    testset_path = Path(config["testset_dir"]) / method_name
    testset_path.mkdir(parents=True, exist_ok=True)
    np.save(testset_path / f"test_indices_run{RUN_ID}_fold{fold_id}.npy", val_indices)
    test_labels = data_views_te.get_all_labels()
    np.save(testset_path / f"test_labels_run{RUN_ID}_fold{fold_id}.npy", test_labels)
    test_ids = data_views_te.get_all_identifiers()
    np.save(testset_path / f"test_identifiers_run{RUN_ID}_fold{fold_id}.npy", test_ids)
    test_indices_file = str(testset_path / f"test_indices_run{RUN_ID}_fold{fold_id}.npy")

    print(f"[sigma={sigma} fold={fold_id}] training...", flush=True)
    set_noise_context(sigma, fold_id)
    method, trainer = MultiFusion_train(
        data_views_tr, val_data=data_views_va, run_id=RUN_ID, fold_id=fold_id,
        method_name=method_name, **config,
    )

    weights_path = Path(config["weights_dir"]) / method_name
    weights_path.mkdir(parents=True, exist_ok=True)
    weights_file = weights_path / f"model_run{RUN_ID}_fold{fold_id}.pt"
    torch.save({
        "run_id": RUN_ID,
        "fold_id": fold_id,
        "model_state_dict": method.state_dict(),
        "best_score": trainer.callbacks[0].best_score.cpu() if hasattr(trainer, "callbacks") else None,
        "epoch": trainer.callbacks[0].stopped_epoch if hasattr(trainer, "callbacks") else None,
        "config": config,
        "test_indices_file": test_indices_file,
    }, weights_file)

    # ── Reload explicitly from checkpoint for Shapley/PS evaluation ──────────
    checkpoint = torch.load(weights_file, map_location="cpu")
    ckpt_config = checkpoint["config"]
    set_noise_context(sigma, fold_id)
    eval_method = build_model(checkpoint, ckpt_config)
    eval_data_te = load_test_data(checkpoint, ckpt_config)
    eval_method.set_missing_info(None, **ckpt_config["training"].get("missing_method", {}))

    y_true = eval_data_te.get_all_labels()
    with torch.no_grad():
        out_full = eval_method.transform(
            create_dataloader(eval_data_te, batch_size=BS, train=False),
            out_norm=output_name(task_type),
            not_return_repre=True,
        )
    full_metrics = compute_metrics(y_true, out_full["prediction"], task_type)
    baseline_metrics = get_random_baseline_metrics(y_true, task_type)

    v_dict = run_subset_inference(
        eval_method, eval_data_te, y_true, view_names, task_type, BS,
        baseline_metrics, full_metrics,
    )
    shapley_per_metric = compute_shapley_per_metric(v_dict, view_names, [METRIC_NAME])
    phi = shapley_per_metric[METRIC_NAME]

    ps_rows = compute_perceptual_score_fold(
        eval_method, eval_data_te, y_true, view_names, view_names, task_type, BS,
        f1_all=full_metrics[METRIC_NAME], n_perm=N_PERM,
        seed=PS_SEED_OFFSET + fold_id, metric_name=METRIC_NAME, verbose=False,
    )
    ps = {row["modality"]: row["PS_raw"] for row in ps_rows}

    sigma_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"modality": v, "phi": phi[v]} for v in view_names]).to_csv(
        sigma_dir / f"shapley_run{RUN_ID}_fold{fold_id}.csv", index=False)
    pd.DataFrame(ps_rows).to_csv(sigma_dir / f"perceptual_score_run{RUN_ID}_fold{fold_id}.csv", index=False)

    print(f"[sigma={sigma} fold={fold_id}] shapley+PS done", flush=True)
    return [{"sigma": sigma, "fold": fold_id, "modality": v, "phi": phi[v], "ps": ps[v]} for v in view_names]


# ── Aggregation + plot ─────────────────────────────────────────────────────────
def aggregate_and_plot(rows: list):
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["sigma", "modality"])
          .agg(phi_mean=("phi", "mean"), phi_std=("phi", "std"),
               ps_mean=("ps", "mean"), ps_std=("ps", "std"))
          .reset_index()
    )
    summary["phi_std"] = summary["phi_std"].fillna(0.0)
    summary["ps_std"] = summary["ps_std"].fillna(0.0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "noise_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}", flush=True)

    fig, (ax_phi, ax_ps) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for modality, color in MODALITY_COLORS.items():
        sub = summary[summary["modality"] == modality].sort_values("sigma")
        if sub.empty:
            continue
        ax_phi.plot(sub["sigma"], sub["phi_mean"], color=color, marker="o", label=modality)
        ax_phi.fill_between(sub["sigma"], sub["phi_mean"] - sub["phi_std"],
                             sub["phi_mean"] + sub["phi_std"], color=color, alpha=0.2)
        ax_ps.plot(sub["sigma"], sub["ps_mean"], color=color, marker="o", label=modality)
        ax_ps.fill_between(sub["sigma"], sub["ps_mean"] - sub["ps_std"],
                            sub["ps_mean"] + sub["ps_std"], color=color, alpha=0.2)

    ax_phi.set_xlabel("sigma")
    ax_phi.set_ylabel(f"Shapley value ({METRIC_NAME})")
    ax_phi.set_title("Shapley value vs. noise")
    ax_ps.set_xlabel("sigma")
    ax_ps.set_title("Perceptual Score vs. noise")
    ax_ps.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig_png = OUT_ROOT / "fig_noise_ablation.png"
    fig_pdf = OUT_ROOT / "fig_noise_ablation.pdf"
    fig.savefig(fig_png, dpi=200)
    fig.savefig(fig_pdf, dpi=200)
    plt.close(fig)
    print(f"Figure saved -> {fig_png} / {fig_pdf}", flush=True)


def main():
    t0 = time.time()
    with open(SETTINGS_FILE) as f:
        base_config = yaml.load(f, Loader=yaml.SafeLoader)

    input_dir_folder = base_config["input_dir_folder"]
    data_name = base_config["data_name"]
    kfolds = base_config["experiment"].get("kfolds", 5)
    r_seed = base_config["experiment"]["runs_seed"][0]

    data_views_tr = load_structure(
        input_dir_folder, data_name,
        load_memory=base_config.get("load_memory", False),
        views_used=base_config["experiment"]["preprocess"]["view_names"],
    )
    data_views_tr.load_stats(input_dir_folder, data_name)
    pristine_identifiers = np.array(data_views_tr.get_all_identifiers(), copy=True)

    if "loss_args" not in base_config["training"]:
        base_config["training"]["loss_args"] = {}
    task = base_config.get("task_type", "").lower()
    if task == "classification":
        base_config["training"]["loss_args"].setdefault("name", "ce")
    elif task == "multilabel":
        base_config["training"]["loss_args"].setdefault("name", "bce")

    method_name = assign_multifusion_name(
        base_config["training"], base_config["method"],
        more_info_str=base_config.get("additional_method_name", ""),
    )

    all_rows = []
    for sigma in SIGMAS:
        # Re-derive the fold split from the pristine identifier order every
        # sigma so splits are bit-for-bit identical across the sweep.
        idx_copy = pristine_identifiers.copy()
        np.random.seed(r_seed)
        np.random.shuffle(idx_copy)
        indexs_runs = np.array_split(idx_copy, kfolds)

        config = build_config(base_config, sigma)

        for fold_id in range(kfolds):
            rows = run_fold(config, sigma, fold_id, data_views_tr, indexs_runs[fold_id], method_name)
            all_rows.extend(rows)

    aggregate_and_plot(all_rows)
    print(f"Total wall-clock time: {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
