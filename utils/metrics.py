"""
Evaluation metrics for classification and multilabel tasks.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, task_type: str) -> dict:
    """
    Compute standard classification metrics from predicted probabilities.

    Args:
        y_true:       Ground-truth labels, shape (N,) or (N, C) for multilabel.
        y_pred_proba: Predicted probabilities, same shape as y_true.
        task_type:    "classification" | "multilabel".

    Returns:
        Dict with keys: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro.
    """
    task = (task_type or "").lower()
    if task == "multilabel":
        y_pred = (y_pred_proba > 0.5).astype(int)
        print("Warning: using 0.5 threshold for multilabel classification.")
    else:
        y_pred = y_pred_proba.argmax(axis=-1)

    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1_macro":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
    }


def get_random_baseline_metrics(y_true: np.ndarray, task_type: str) -> dict:
    """
    Expected metrics for a random classifier (analytical derivation).

    Used as the empty-coalition value v(∅) in Shapley computations.

    Args:
        y_true:    Ground-truth labels, shape (N,) or (N, C).
        task_type: "classification" | "multilabel".

    Returns:
        Dict with same keys as compute_metrics.
    """
    task = (task_type or "").lower()

    if task == "multilabel":
        return {
            "accuracy": 0.5,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
        }

    unique_classes = np.unique(y_true)

    if len(unique_classes) == 2:
        p_pos = np.mean(y_true)
        p_neg = 1 - p_pos

        tp = p_pos * p_pos
        fn = p_pos * p_neg
        fp = p_neg * p_pos
        tn = p_neg * p_neg

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = tp + tn

        return {
            "accuracy":        accuracy,
            "f1_macro":        f1,
            "f1_weighted":     f1,
            "precision_macro": precision,
            "recall_macro":    recall,
        }

    elif len(unique_classes) > 2:
        random_acc = 1.0 / len(unique_classes)
        return {
            "accuracy":        random_acc,
            "f1_macro":        random_acc,
            "f1_weighted":     random_acc,
            "precision_macro": random_acc,
            "recall_macro":    random_acc,
        }

    # Fallback (single class)
    return {
        "accuracy": 0.5,
        "f1_macro": 0.5,
        "f1_weighted": 0.5,
        "precision_macro": 0.5,
        "recall_macro": 0.5,
    }