"""
shap_analysis/perceptual_score.py
----------------------------------
Permutation-based Perceptual Score (Gat et al., NeurIPS 2021) for
multi-sensor/multi-view models.

PS_raw(m)   = F1(all) - mean_over_perms F1(all | m permuted)
PS_task(m)  = PS_raw(m) / (1 - accuracy_majority_vote)
PS_model(m) = PS_raw(m) / F1(all)

Permutation is applied at the identifier->index mapping level
(views_data_ident2indx), so that a sample's modality m is replaced by the
modality m of a different (permuted) sample, while every other modality and
the label stay aligned to the original sample.
"""
import numpy as np

from code.datasets.utils import create_dataloader
from code.training.utils import output_name
from utils.metrics import compute_metrics


def _raw_views_for(modality: str) -> list:
    """Composite view names (e.g. 'S2_S2VI') map to several raw xarray views."""
    return modality.split("_") if "_" in modality else [modality]


def permute_modality_identifiers(data_te, modality: str, perm_idx: np.ndarray) -> dict:
    """
    Permute the identifier->index mapping for a modality's raw views, restricted
    to the current val_identifiers. Returns the original mappings for restoration.
    """
    raw_views = _raw_views_for(modality)
    val_ids   = data_te.val_identifiers
    backups   = {}

    for raw in raw_views:
        orig_map = data_te.views_data_ident2indx[raw]
        backups[raw] = orig_map
        new_map = dict(orig_map)
        for i, ident in enumerate(val_ids):
            new_map[ident] = orig_map[val_ids[perm_idx[i]]]
        data_te.views_data_ident2indx[raw] = new_map

    return backups


def restore_modality_identifiers(data_te, backups: dict):
    for raw, orig_map in backups.items():
        data_te.views_data_ident2indx[raw] = orig_map


def compute_perceptual_score_fold(
    method,
    data_te,
    y_true: np.ndarray,
    view_names: list,
    modalities: list,
    task_type: str,
    batch_size: int,
    f1_all: float,
    n_perm: int = 10,
    seed: int = 0,
    metric_name: str = "f1_weighted",
    verbose: bool = True,
) -> list:
    """
    Compute the raw Perceptual Score for each modality in `modalities`.

    Args:
        method:      Trained model (eval mode).
        data_te:     Test Dataset_MultiView (val_identifiers set).
        y_true:      Ground-truth labels, aligned with data_te's current order.
        view_names:  Full view list passed to the model on every forward pass
                     (includes any fixed views, e.g. 'geo' — never permuted
                     unless explicitly listed in `modalities`).
        modalities:  Views to permute (free/explainable sensors).
        f1_all:      Metric value of the unpermuted full-coalition forward pass.
        n_perm:      Number of random permutations per modality.
        metric_name: Metric key used to compute PS_raw (default f1_weighted).

    Returns:
        List of dicts: {modality, f1_all, f1_permuted_mean, f1_permuted_std, PS_raw, n_perm}
    """
    rng   = np.random.default_rng(seed)
    n_val = len(data_te.val_identifiers)
    rows  = []

    for modality in modalities:
        perm_metrics = []
        for p in range(n_perm):
            perm_idx = rng.permutation(n_val)
            backups  = permute_modality_identifiers(data_te, modality, perm_idx)
            try:
                out = method.transform(
                    create_dataloader(data_te, batch_size=batch_size, train=False),
                    out_norm=output_name(task_type),
                    args_forward={"inference_views": view_names,
                                  "missing_method": method.missing_method},
                    perc_forward=1.0,
                    not_return_repre=True,
                )
                m = compute_metrics(y_true, out["prediction"], task_type)
            finally:
                restore_modality_identifiers(data_te, backups)
            perm_metrics.append(m[metric_name])

        perm_mean = float(np.mean(perm_metrics))
        perm_std  = float(np.std(perm_metrics))
        ps_raw    = f1_all - perm_mean

        if verbose:
            print(f"    {modality:<10} F1(all)={f1_all:.4f}  "
                  f"F1(permuted)={perm_mean:.4f}+/-{perm_std:.4f}  "
                  f"PS_raw={ps_raw:.4f}", flush=True)

        rows.append({
            "modality": modality,
            "f1_all": f1_all,
            "f1_permuted_mean": perm_mean,
            "f1_permuted_std": perm_std,
            "PS_raw": ps_raw,
            "n_perm": n_perm,
        })

    return rows
