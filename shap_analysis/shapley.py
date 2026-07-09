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
from .characteristic import char_value, baseline_metrics_all, DEFAULT_METRIC


# ── v-dict access (format v1 / v2) ────────────────────────────────────────────
#
# v1 (legacy, on disk):  {frozenset(S) -> metrics_dict}
# v2 (current):          {"format": "v2", "y_true": …, "task_type": …,
#                         "coalitions": {frozenset(S) -> record}}
#   record = {"pred": proba_NxC}                for a computed coalition
#          | {"metrics": {...}}                 for the analytical baseline (∅ / fixed)
#
# All consumers below go through scalar_vdict(), so both formats work
# transparently. Only v2 (predictions stored) supports swapping the
# characteristic function as a post-processing flag; v1 stays pinned to the
# metrics that were computed at inference time.

def is_v2(v_dict: dict) -> bool:
    """True if v_dict is the prediction-storing v2 format."""
    return isinstance(v_dict, dict) and "coalitions" in v_dict


def coalitions(v_dict: dict) -> dict:
    """Return the {frozenset -> record} mapping regardless of format."""
    return v_dict["coalitions"] if is_v2(v_dict) else v_dict


def coalition_scalar(record, metric: str, y_true=None, task_type: str = "classification") -> float:
    """
    Collapse a single coalition record to the scalar v(S) under ``metric``.

    v2 computed coalition -> evaluate the characteristic function on stored preds.
    v2 baseline coalition -> read the stored analytical metric.
    v1 record (plain metrics dict) -> read the stored scalar directly.
    """
    if isinstance(record, dict) and "pred" in record:
        return char_value(y_true, record["pred"], metric, task_type)
    # v2 baseline coalition, or v1 record (which IS the metrics dict).
    metrics = record["metrics"] if isinstance(record, dict) and "metrics" in record else record
    if metric not in metrics:
        raise KeyError(
            f"metric '{metric}' is not stored in this (legacy v1) v-dict "
            f"(available: {sorted(metrics)}). Regenerate the v-dicts in v2 format "
            f"(scripts.vdict.build_vdicts --shapley) to compute new characteristic "
            f"functions like MCC/balanced_accuracy from stored predictions.")
    return float(metrics[metric])


def scalar_vdict(v_dict: dict, metric: str) -> dict:
    """
    Collapse a v-dict (v1 or v2) to {frozenset -> scalar v(S)} under ``metric``.

    This is the single entry point every downstream computation uses, so the
    characteristic function is chosen in exactly one place.
    """
    y_true    = v_dict.get("y_true") if is_v2(v_dict) else None
    task_type = v_dict.get("task_type", "classification") if is_v2(v_dict) else "classification"
    return {s: coalition_scalar(rec, metric, y_true, task_type)
            for s, rec in coalitions(v_dict).items()}


# ── Core formula ──────────────────────────────────────────────────────────────

def shapley_values(view_names: list, v_dict: dict, fixed_views: list = None) -> dict:
    """
    Exact Shapley values via the cooperative game theory formula.

    Args:
        view_names:  Player names (sensor/modality names).
        v_dict:      frozenset → scalar. Must contain frozenset() (empty coalition),
                     or frozenset(fixed_views) when fixed_views is given.
        fixed_views: Views always present in every coalition — not explained
                     (phi=0), but implicitly included in every v_dict key looked up.

    Returns:
        {view_name: shapley_value}. Fixed views get phi=0.
    """
    fixed_views = fixed_views or []
    free_views  = [v for v in view_names if v not in fixed_views]
    fixed_set   = frozenset(fixed_views)
    n_free      = len(free_views)

    shapley = {}
    for view in free_views:
        others = [v for v in free_views if v != view]
        phi = 0.0
        for size in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, size):
                S_free = frozenset(S_tuple)
                S = S_free | fixed_set
                s = len(S_free)
                weight = math.factorial(s) * math.factorial(n_free - s - 1) / math.factorial(n_free)
                phi += weight * (v_dict[S | {view}] - v_dict[S])
        shapley[view] = phi

    for view in fixed_views:
        shapley[view] = 0.0

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
    fixed_views: list = None,
    full_pred: np.ndarray = None,
) -> dict:
    """
    Run inference for all strict subsets (size 1 … N-1) and build the v2 v_dict.

    The v2 format stores the raw per-coalition predictions (softmax probabilities),
    NOT just a scalar metric. This is the key architectural point: with predictions
    stored, the characteristic function (f1_weighted, MCC, balanced accuracy, …)
    becomes a post-processing flag (see characteristic.py / scalar_vdict) instead of
    requiring a fresh round of model inference.

    Args:
        method:           Trained model.
        data_te:          Test Dataset_MultiView.
        y_true:           Ground-truth labels.
        view_names:       Full list of view/sensor names.
        task_type:        "classification" | "multilabel".
        batch_size:       Inference batch size.
        baseline_metrics: Analytical metrics for the empty coalition v(∅). Enriched
                          with MCC / balanced_accuracy so v(∅) is metric-parametrizable.
        full_metrics:     Metrics for the full coalition v(N); used only as a fallback
                          if ``full_pred`` is not supplied.
        verbose:          Print progress for each subset.
        fixed_views:      Views always present in every coalition (not explained, but
                          kept in every forward pass). The fixed-only coalition is the
                          baseline (no information added).
        full_pred:        Softmax predictions of the full coalition v(N). Pass this so
                          v(N) is metric-parametrizable like every other coalition.

    Returns:
        v_dict (v2): {"format": "v2", "y_true": …, "task_type": …,
                      "coalitions": {frozenset(S) -> record}} with ∅ and N included.
    """
    fixed_views = fixed_views or []
    free_views  = [v for v in view_names if v not in fixed_views]
    n_free      = len(free_views)

    # Analytical baseline (all metrics incl. MCC / balanced_accuracy) for v(∅).
    baseline_all = {**baseline_metrics_all(y_true, task_type), **(baseline_metrics or {})}
    baseline_rec = {"metrics": baseline_all}
    full_rec = {"pred": np.asarray(full_pred)} if full_pred is not None else {"metrics": full_metrics}

    coalition_records = {
        frozenset():           baseline_rec,
        frozenset(view_names): full_rec,
    }
    if fixed_views:
        coalition_records[frozenset(fixed_views)] = baseline_rec

    n_subsets = sum(math.comb(n_free, s) for s in range(1, n_free))
    done = 0

    for size in range(1, n_free):
        for subset_tuple in itertools.combinations(free_views, size):
            subset_list = list(subset_tuple) + fixed_views
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
            coalition_records[frozenset(subset_list)] = {"pred": np.asarray(out_sub["prediction"])}
            done += 1

            if verbose:
                label = "+".join(subset_list)
                f1w = char_value(y_true, out_sub["prediction"], "f1_weighted", task_type)
                print(f"  [{done}/{n_subsets}] {label} -> f1_weighted={f1w:.4f}", flush=True)

    return {
        "format":     "v2",
        "y_true":     np.asarray(y_true),
        "task_type":  task_type,
        "view_names": list(view_names),
        "coalitions": coalition_records,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def compute_shapley_per_metric(
    v_dict: dict,
    view_names: list,
    metric_keys: list,
    fixed_views: list = None,
) -> dict:
    """
    Compute Shapley values for each metric from a pre-built v_dict.

    Args:
        v_dict:      {frozenset → metrics_dict} (output of run_subset_inference).
        view_names:  Full list of view names.
        metric_keys: Metrics to compute Shapley values for.
        fixed_views: Views always present in every coalition (phi=0 for these).

    Returns:
        {metric_name: {view_name: shapley_value}}
    """
    result = {}
    for metric_name in metric_keys:
        v_scalar = scalar_vdict(v_dict, metric_name)
        result[metric_name] = shapley_values(view_names, v_scalar, fixed_views=fixed_views)
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
        v = scalar_vdict(v_dict, metric_name)
        sii = {}
        for i, vi in enumerate(view_names):
            for j, vj in enumerate(view_names):
                if j <= i:
                    continue
                others = [v_ for v_ in view_names if v_ not in (vi, vj)]
                phi_ij = 0.0
                for size in range(len(others) + 1):
                    for S_tuple in itertools.combinations(others, size):
                        S      = frozenset(S_tuple)
                        s      = len(S)
                        weight = (math.factorial(s) * math.factorial(n - s - 2)
                                  / math.factorial(n - 1))
                        delta  = (v[S | {vi, vj}]
                                  - v[S | {vi}]
                                  - v[S | {vj}]
                                  + v[S])
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

    # Raw subset metrics (stored for traceability) — recomputed per metric from v2.
    for metric_name in metric_keys:
        for fs, val in scalar_vdict(v_dict, metric_name).items():
            if fs == frozenset():
                continue
            combi_str = "_".join(sorted(fs))
            col = f"shapley_{combi_str}_{metric_name}"
            columns.setdefault(col, []).append(val)

    # Shapley values per metric
    shapley_per_metric = compute_shapley_per_metric(v_dict, view_names, metric_keys)
    for metric_name, sv in shapley_per_metric.items():
        for view, phi in sv.items():
            col = f"shapley_value_{view}_{metric_name}"
            columns.setdefault(col, []).append(phi)

    return columns