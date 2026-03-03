"""
Evaluation metrics utilities.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ──────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like  – ground-truth labels (0 or 1).
    y_pred : array-like  – predicted labels.
    y_prob : array-like, optional – predicted probabilities for the positive class.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            result["roc_auc"] = float("nan")
    else:
        result["roc_auc"] = float("nan")
    return result


def get_roc_curve(y_true, y_prob):
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)


# ──────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────

def results_to_dataframe(results_dict):
    """
    Convert a nested results dictionary to a tidy DataFrame.

    Parameters
    ----------
    results_dict : dict
        Nested dict: {dataset: {strategy: {model: metrics_dict}}}

    Returns
    -------
    pd.DataFrame with columns: dataset, strategy, model, accuracy, precision,
                                recall, f1, roc_auc.
    """
    rows = []
    for ds, strats in results_dict.items():
        for strat, models in strats.items():
            for model, metrics in models.items():
                row = {
                    "dataset": ds,
                    "strategy": strat,
                    "model": model,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                }
                rows.append(row)
    return pd.DataFrame(rows)

