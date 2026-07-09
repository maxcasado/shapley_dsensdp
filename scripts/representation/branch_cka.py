"""
compute_branch_cka.py
----------------------
Centered Kernel Alignment (CKA) between the per-modality encoder branches
of a trained FCoM model.

FCoM builds one encoder "branch" per view/modality (e.g. S2_S2VI, S1,
weather, DEM) before fusing them into a joint prediction (see
code/models/base_fusion.py::MVFusionMissing.view_encoders). This script
loads one or more trained checkpoints, runs the test set through every
branch to get its raw representation ("views:rep"), and computes the
pairwise linear CKA similarity between every pair of branches -- a measure
of how much two modality-specific encoders converge to the same
representational geometry.

Linear CKA (Kornblith et al., 2019, "Similarity of Neural Network
Representations Revisited"):

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

with X (N, D_x) and Y (N, D_y) mean-centered along the sample axis
(rows = test samples, aligned by index). CKA = 1 means the two branches
encode the same information up to rotation/isotropic scaling; CKA = 0
means they are uncorrelated.

Where the activations come from (verified against code/models/base_fusion.py):
  - "views:rep" (forward_encoders) is exactly `view_encoders[v](x)`, the same
    tensor forward() lists into `views_data` right before `fusion_module(...)`
    -- i.e. the per-branch representation just before fusion, not a deeper
    (post-fusion) or shallower (raw input) layer.
  - This is only a clean embedding for method.feature=True checkpoints
    (FeatureFusion). For method.feature=False (DecisionFusion) each branch
    ends in its own private prediction head, so "views:rep" is per-class
    logits, not a representation -- the script warns when it detects this.
  - If znorm_logits is enabled, forward() z-scores a *local copy* of
    views_data before fusion without writing it back into "views:rep"; this
    script re-applies that same normalization so CKA always matches what
    fusion actually consumes.

Pooling / shape homogeneity across branches:
  - N (rows): identical by construction -- every branch is read off the same
    dataloader with shuffle=False, so row i is the same test sample in every
    branch's array (checked with an assertion below).
  - D (columns): identical (=emb_dim) for every branch because
    Generic_Encoder (code/models/single/base_encoders.py) appends the same
    final nn.Linear(*, emb_dim) projection regardless of the sub-model.
  - Temporal aggregation: S1/S2_S2VI/weather all use TempCNN with the same
    seq_len -- flatten-then-dense pooling identical across those three
    branches. DEM has no time axis (static features) so there is nothing to
    pool; that asymmetry is a property of the data, not of this script.

Usage:
    # Single run/fold
    python -m scripts.representation.branch_cka -s config/com_average.yaml -r 0 -f 0

    # Aggregate CKA (mean +/- std) over all folds of one run
    python -m scripts.representation.branch_cka -s config/com_average.yaml -r 0 --fold_ids 0 1 2 3 4

    # Aggregate over several runs and folds
    python -m scripts.representation.branch_cka -s config/com_average.yaml --run_ids 0 1 --fold_ids 0 1 2 3 4
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from utils.checkpoint import build_model, load_test_data, resolve_weights_path
    from code.datasets.utils import create_dataloader
except Exception:
    print(f"\nImport error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


# Consistent with run_comparison_analysis.py / visualize/maps.py view colors
VIEW_COLORS = {
    "S2_S2VI": "#4e79a7", "S1": "#f28e2b", "weather": "#e15759",
    "DEM": "#76b7b2", "geo": "#59a14f",
}


def log(msg: str):
    print(msg, flush=True)


# ==============================================================================
#  Linear CKA
# ==============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representations of the same N aligned samples.

    X: (N, D_x), Y: (N, D_y) -- rows are samples, columns are features.
    Invariant to orthogonal transform and isotropic scaling of either input.
    """
    # Column (per-feature) centering -- mean over axis=0 (samples), one value
    # per feature. Equivalent to double-centering the Gram matrices (H K H,
    # H K H') before taking HSIC; verified numerically against that formula.
    # Without it CKA mostly measures mean alignment, not shared covariance
    # structure (see e.g. a translation-invariance test). This is a different
    # axis than method._znorm's per-sample (row) normalization applied
    # upstream in get_branch_representations -- the two do not substitute
    # for each other and both can be active at once.
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    cross = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(cross / denom) if denom > 0 else float("nan")


def cka_matrix(reps: dict) -> pd.DataFrame:
    """Pairwise linear CKA between every pair of branches in `reps`."""
    views = list(reps.keys())
    mat = np.eye(len(views))
    for i, vi in enumerate(views):
        for j, vj in enumerate(views):
            if j < i:
                continue
            mat[i, j] = mat[j, i] = linear_cka(reps[vi], reps[vj])
    return pd.DataFrame(mat, index=views, columns=views)


# ==============================================================================
#  Per-checkpoint branch representations
# ==============================================================================

def get_branch_representations(run_id: int, fold_id: int, config: dict,
                                weights_file: str, batch_size: int):
    """Load a checkpoint and return {view_name: (N, D) np.ndarray} on its test set."""
    weights_path = Path(weights_file) if weights_file else resolve_weights_path(config, run_id, fold_id)
    if not weights_path.exists():
        log(f"  WARNING: checkpoint not found, skipping: {weights_path}")
        return None

    log(f"  Checkpoint: {weights_path}")
    checkpoint  = torch.load(weights_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)

    # FCoM has two fusion modes (code/training/learn_pipeline.py::build_model):
    # method.feature=True  -> FeatureFusion, each branch is a Generic_Encoder,
    #                         view_encoders[v](x) is a (N, emb_dim) embedding.
    # method.feature=False -> DecisionFusion, each branch is
    #                         Sequential(encoder, private_prediction_head), so
    #                         view_encoders[v](x) is (N, n_classes) *logits*,
    #                         not an embedding. CKA on that answers "do branches
    #                         predict alike", not "do branches share structure",
    #                         and is close to degenerate at n_classes=2.
    if not ckpt_config.get("method", {}).get("feature", True):
        log("  WARNING: this checkpoint is a decision-level fusion (method.feature=False) -- "
            "'views:rep' here is each branch's own prediction logits, not an embedding. "
            "CKA on this measures decision agreement, not representational redundancy.")

    method  = build_model(checkpoint, ckpt_config)
    data_te = load_test_data(checkpoint, ckpt_config)

    bs = batch_size or ckpt_config["training"]["batch_size"]
    # shuffle=False (train=False) so every view is read off the same sample
    # order -> row i is the same test point in every branch's array below.
    loader = create_dataloader(data_te, batch_size=bs, train=False)

    out = method.transform(loader, intermediate=True, not_return_repre=False)
    reps = out["views:rep"]  # {view_name: (N, D) np.ndarray}, N = len(data_te)
    reps = {v: np.asarray(reps[v]) for v in reps}

    # "views:rep" is always the *raw* per-branch encoder output (see
    # base_fusion.py::forward): when znorm_logits is enabled, forward()
    # z-scores a local copy before handing it to fusion_module but never
    # writes that back into the returned dict. Re-apply the exact same
    # (static) method here so CKA runs on what fusion actually receives.
    if getattr(method, "znorm_logits", False):
        log("  znorm_logits enabled on this checkpoint -- re-applying the same "
            "per-sample normalization used just before fusion.")
        reps = {v: method._znorm(torch.from_numpy(r)).numpy() for v, r in reps.items()}

    n_samples = {v: r.shape[0] for v, r in reps.items()}
    if len(set(n_samples.values())) != 1:
        raise RuntimeError(f"Branches have inconsistent sample counts, CKA would be invalid: {n_samples}")
    for v, r in reps.items():
        log(f"    {v}: {r.shape}")

    return reps


# ==============================================================================
#  Plot
# ==============================================================================

def plot_cka_heatmap(df: pd.DataFrame, out_path: Path, title: str = ""):
    views = list(df.columns)
    n = len(views)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1.5))
    im = ax.imshow(df.values, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(views, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(views)
    for i, v in enumerate(views):
        color = VIEW_COLORS.get(v, "#333333")
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)

    for i in range(n):
        for j in range(n):
            value = df.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     color="white" if value > 0.6 else "black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
#  Main
# ==============================================================================

def main(args):
    config = yaml.safe_load(Path(args.settings_file).read_text())

    run_ids  = args.run_ids if args.run_ids is not None else [args.run_id]
    fold_ids = args.fold_ids if args.fold_ids is not None else [args.fold_id if args.fold_id is not None else 0]

    single_checkpoint = len(run_ids) == 1 and len(fold_ids) == 1
    method_name = resolve_weights_path(config, run_ids[0], fold_ids[0]).parent.name
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(config.get("output_dir_folder", ".")) / "branch_cka" / method_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_matrices = []
    view_names = None
    for run_id in run_ids:
        for fold_id in fold_ids:
            log(f"\n{'─' * 60}\n  Run {run_id} / Fold {fold_id}\n{'─' * 60}")
            weights_file = args.weights_file if (args.weights_file and single_checkpoint) else None
            reps = get_branch_representations(run_id, fold_id, config, weights_file, args.batch_size)
            if reps is None:
                continue

            mat = cka_matrix(reps)
            if view_names is None:
                view_names = list(mat.columns)
            mat = mat.loc[view_names, view_names]
            per_fold_matrices.append(mat)
            log(f"  CKA matrix:\n{mat.round(3).to_markdown()}")

    if len(per_fold_matrices) == 0:
        log("No checkpoints found -- nothing to do.")
        return

    stacked = np.stack([m.values for m in per_fold_matrices], axis=0)
    df_mean = pd.DataFrame(stacked.mean(axis=0), index=view_names, columns=view_names)
    df_std  = pd.DataFrame(stacked.std(axis=0), index=view_names, columns=view_names)

    df_mean.to_csv(out_dir / "cka_mean.csv")
    df_std.to_csv(out_dir / "cka_std.csv")

    log(f"\n################ CKA between branches (mean over {len(per_fold_matrices)} run/fold) ################")
    log(df_mean.round(3).to_markdown())
    log("\n################ std ################")
    log(df_std.round(3).to_markdown())

    plot_cka_heatmap(df_mean, out_dir / "cka_heatmap.pdf", title=f"Branch CKA -- {method_name}")

    log(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Linear CKA between FCoM per-modality branches.")
    p.add_argument("--settings_file", "-s", required=True, help="Path to the experiment config yaml.")
    p.add_argument("--weights_file", "-w", default=None,
                   help="Explicit checkpoint path (only used for a single run/fold).")
    p.add_argument("--run_id", "-r", type=int, default=0)
    p.add_argument("--run_ids", type=int, nargs="+", default=None,
                   help="Aggregate CKA over several runs (e.g. --run_ids 0 1 2).")

    fold_group = p.add_mutually_exclusive_group()
    fold_group.add_argument("--fold_id", "-f", type=int, default=None)
    fold_group.add_argument("--fold_ids", type=int, nargs="+", default=None,
                             help="Aggregate CKA over several folds (e.g. --fold_ids 0 1 2 3 4).")

    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--out_dir", "-o", default=None,
                   help="Output dir (default: <output_dir_folder>/branch_cka/<method_name>).")
    args = p.parse_args()

    main(args)
