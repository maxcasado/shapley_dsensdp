"""
shap_analysis/characteristic.py
-------------------------------
Characteristic-function registry for the cooperative game.

This is what makes the characteristic function v(S) a *post-processing flag*
rather than something frozen at inference time. Once the v-dicts store the
per-coalition predictions (format v2, see shapley.py::run_subset_inference),
swapping f1_weighted for MCC or balanced accuracy is a one-line change here —
no model re-inference required.

Public API
----------
CHAR_FUNCS       : {name -> fn(y_true, y_pred_proba, task_type) -> float}.
char_value       : evaluate one characteristic function on stored predictions.
baseline_value   : analytical v(∅) for the random/prior classifier, per metric.
DEFAULT_METRIC   : "f1_weighted" (the paper's reported characteristic function).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score,
)

from utils.metrics import get_random_baseline_metrics

DEFAULT_METRIC = "f1_weighted"


def _labels(y_pred_proba: np.ndarray, task_type: str) -> np.ndarray:
    """Hard labels from stored probabilities (argmax, or 0.5 threshold for multilabel)."""
    if (task_type or "").lower() == "multilabel":
        return (y_pred_proba > 0.5).astype(int)
    return np.asarray(y_pred_proba).argmax(axis=-1)


# Each entry: (y_true, y_pred_proba, task_type) -> scalar. Kept in lock-step with
# utils.metrics.compute_metrics for the five shared metrics, extended with MCC and
# balanced accuracy (the alternative characteristic functions from the manifesto).
CHAR_FUNCS = {
    "accuracy":          lambda yt, yp, tt: accuracy_score(yt, _labels(yp, tt)),
    "f1_macro":          lambda yt, yp, tt: f1_score(yt, _labels(yp, tt), average="macro", zero_division=0),
    "f1_weighted":       lambda yt, yp, tt: f1_score(yt, _labels(yp, tt), average="weighted", zero_division=0),
    "precision_macro":   lambda yt, yp, tt: precision_score(yt, _labels(yp, tt), average="macro", zero_division=0),
    "recall_macro":      lambda yt, yp, tt: recall_score(yt, _labels(yp, tt), average="macro", zero_division=0),
    "mcc":               lambda yt, yp, tt: matthews_corrcoef(yt, _labels(yp, tt)),
    "balanced_accuracy": lambda yt, yp, tt: balanced_accuracy_score(yt, _labels(yp, tt)),
}


def char_value(y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str,
               task_type: str = "classification") -> float:
    """Evaluate characteristic function ``metric`` on stored predictions."""
    if metric not in CHAR_FUNCS:
        raise KeyError(f"unknown characteristic function '{metric}'; "
                       f"available: {sorted(CHAR_FUNCS)}")
    return float(CHAR_FUNCS[metric](y_true, y_pred_proba, task_type))


def baseline_metrics_all(y_true: np.ndarray, task_type: str = "classification") -> dict:
    """
    Full analytical baseline v(∅) dict covering every metric in CHAR_FUNCS.

    Extends utils.metrics.get_random_baseline_metrics (the five original metrics,
    derived for a prior-matching random classifier) with the two additional
    characteristic functions:
        MCC                -> 0.0  (a random classifier is uncorrelated with y)
        balanced_accuracy  -> 0.5  (chance level, invariant to class imbalance)
    """
    base = dict(get_random_baseline_metrics(y_true, task_type))
    base.setdefault("mcc", 0.0)
    base.setdefault("balanced_accuracy", 0.5)
    return base


def baseline_value(y_true: np.ndarray, metric: str,
                   task_type: str = "classification") -> float:
    """Analytical empty-coalition value v(∅) for a single metric."""
    return float(baseline_metrics_all(y_true, task_type)[metric])
