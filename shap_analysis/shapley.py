"""
Shapley value computation for multi-sensor/multi-view models.

Public API
----------
shapley_values                      : pure cooperative game theory formula (scalar v_dict).
run_subset_inference                : run all 2^N-1 subset inferences, build metric v_dict.
compute_shapley_per_metric          : aggregate v_dict → {metric: {view: phi}}.
compute_shapley_interactions_metrics: SII order-2 from a metric-based v_dict.
compute_shapley_fold                : end-to-end helper for train_multi; returns flat dict
                                      ready to extend metadata_r.
"""
import math
import itertools

import numpy as np

from code.datasets.utils import create_dataloader
from code.training.utils import output_name
from utils.metrics import compute_metrics, get_random_baseline_metrics


# ── Core formula ──────────────────────────────────────────────────────────────

def shapley_values(view_names: list, v_dict: dict) -> dict:
    """
    Exact Shapley values via the cooperative game theory formula.

    Args:
        view_names: Player names (sensor/modality names).
        v_dict:     frozenset → scalar. Must contain frozenset() (empty coalition).

    Returns:
        {view_name: shapley_value}
    """
    n = len(view_names)
    v_empty = v_dict.get(frozenset(), 0)
    shapley = {}

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


# ── Subset inference ──────────────────────────────────────────────────────────

def run_subset_inference(
    method,
    data_te,
    y_true: np.ndarray,
    view_names: list,
    task_type: str,
    batch_size: int,
    baseline_metrics: dict,
    full_metrics: dict,
    verbose: bool = False,
) -> dict:
    """
    Run inference for all strict subsets (size 1 … N-1) and build the v_dict.

    Args:
        method:           Trained model.
        data_te:          Test Dataset_MultiView.
        y_true:           Ground-truth labels.
        view_names:       Full list of view/sensor names.
        task_type:        "classification" | "multilabel".
        batch_size:       Inference batch size.
        baseline_metrics: Metrics for the empty coalition v(∅).
        full_metrics:     Metrics for the full coalition v(N).
        verbose:          Print progress for each subset.

    Returns:
        v_dict: {frozenset → metrics_dict} for all coalitions including ∅ and N.
    """
    v_dict = {
        frozenset():           baseline_metrics,
        frozenset(view_names): full_metrics,
    }

    n_subsets = sum(math.comb(len(view_names), s) for s in range(1, len(view_names)))
    done = 0

    for size in range(1, len(view_names)):
        for subset_tuple in itertools.combinations(view_names, size):
            subset_list = list(subset_tuple)
            args_fwd = {
                "inference_views": subset_list,
                "missing_method":  method.missing_method,
            }
            out_sub = method.transform(
                create_dataloader(data_te, batch_size=batch_size, train=False),
                out_norm=output_name(task_type),
                args_forward=args_fwd,
                perc_forward=1.0,
                not_return_repre=True,
            )
            sub_metrics = compute_metrics(y_true, out_sub["prediction"], task_type)
            v_dict[frozenset(subset_list)] = sub_metrics
            done += 1

            if verbose:
                label = "+".join(subset_list)
                print(f"  [{done}/{n_subsets}] {label} -> f1_macro={sub_metrics['f1_macro']:.4f}", flush=True)

    return v_dict


# ── Aggregation ───────────────────────────────────────────────────────────────

def compute_shapley_per_metric(
    v_dict: dict,
    view_names: list,
    metric_keys: list,
) -> dict:
    """
    Compute Shapley values for each metric from a pre-built v_dict.

    Args:
        v_dict:      {frozenset → metrics_dict} (output of run_subset_inference).
        view_names:  Full list of view names.
        metric_keys: Metrics to compute Shapley values for.

    Returns:
        {metric_name: {view_name: shapley_value}}
    """
    result = {}
    for metric_name in metric_keys:
        v_scalar = {s: v_dict[s][metric_name] for s in v_dict}
        result[metric_name] = shapley_values(view_names, v_scalar)
    return result


# ── Shapley Interaction Index — metric-based (scalar) ────────────────────────

def compute_shapley_interactions_metrics(
    v_dict: dict,
    view_names: list,
    metric_keys: list,
) -> dict:
    """
    Compute the Shapley Interaction Index (SII) order-2 for all view pairs,
    using scalar metric values as the characteristic function.

    SII(i,j) = Σ_{S ⊆ N\\{i,j}} w(|S|) · Δ_{ij}(S)
    Δ_{ij}(S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
    w(s)      = s! · (n-s-2)! / (n-1)!

    Args:
        v_dict:      {frozenset → metrics_dict} from run_subset_inference.
                     Must contain all 2^N coalitions (∅ included).
        view_names:  Full list of view names.
        metric_keys: Metrics to compute SII for.

    Returns:
        {metric_name: {(view_i, view_j): sii_value}}
        Only the upper triangle is stored (i < j); the index is a tuple of names.
    """
    n      = len(view_names)
    result = {}

    for metric_name in metric_keys:
        sii = {}
        for i, vi in enumerate(view_names):
            for j, vj in enumerate(view_names):
                if j <= i:
                    continue
                others = [v for v in view_names if v not in (vi, vj)]
                phi_ij = 0.0
                for size in range(len(others) + 1):
                    for S_tuple in itertools.combinations(others, size):
                        S      = frozenset(S_tuple)
                        s      = len(S)
                        weight = (math.factorial(s) * math.factorial(n - s - 2)
                                  / math.factorial(n - 1))
                        delta  = (v_dict[S | {vi, vj}][metric_name]
                                  - v_dict[S | {vi}][metric_name]
                                  - v_dict[S | {vj}][metric_name]
                                  + v_dict[S][metric_name])
                        phi_ij += weight * delta
                sii[(vi, vj)] = phi_ij
        result[metric_name] = sii

    return result


# ── End-to-end fold helper (used by train_multi) ──────────────────────────────

def compute_shapley_fold(
    method,
    data_views_te,
    config_file: dict,
    y_true: np.ndarray,
    full_metrics: dict,
    metric_keys: list,
    batch_size: int,
) -> dict:
    """
    Run all subset inferences and return Shapley columns for one fold.

    Returns a flat dict {column_name: [value]} ready to extend metadata_r.
    """
    view_names = config_file["experiment"]["preprocess"]["view_names"]
    task_type  = config_file.get("task_type")

    method.set_missing_info(None, **config_file["training"].get("missing_method", {}))
    baseline_metrics = get_random_baseline_metrics(y_true, task_type)

    v_dict = run_subset_inference(
        method, data_views_te, y_true, view_names, task_type,
        batch_size, baseline_metrics, full_metrics,
    )

    columns = {}

    # Raw subset metrics (stored for traceability)
    for fs, metrics in v_dict.items():
        if fs == frozenset():
            continue
        combi_str = "_".join(sorted(fs))
        for k_m, v_m in metrics.items():
            col = f"shapley_{combi_str}_{k_m}"
            columns.setdefault(col, []).append(v_m)

    # Shapley values per metric
    shapley_per_metric = compute_shapley_per_metric(v_dict, view_names, metric_keys)
    for metric_name, sv in shapley_per_metric.items():
        for view, phi in sv.items():
            col = f"shapley_value_{view}_{metric_name}"
            columns.setdefault(col, []).append(phi)

    return columns