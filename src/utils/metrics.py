"""
Evaluation metrics and statistical comparison utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from scipy import stats
import warnings
import time


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


def get_classification_report(y_true, y_pred):
    """Return a formatted classification report string."""
    return classification_report(y_true, y_pred, zero_division=0)


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


def compare_models(results_df, metric="f1"):
    """
    Produce a pivot table comparing models across datasets and strategies.
    """
    return results_df.pivot_table(
        values=metric, index=["dataset", "strategy"], columns="model"
    )


# ──────────────────────────────────────────────
# Statistical significance tests
# ──────────────────────────────────────────────

def paired_ttest(scores_a, scores_b, alpha=0.05):
    """
    Paired t-test between two sets of cross-validation scores.

    Returns
    -------
    dict with t_statistic, p_value, significant (bool).
    """
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return {"t_statistic": t_stat, "p_value": p_val, "significant": p_val < alpha}


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for two classifiers on the same test set.

    Returns
    -------
    dict with chi2, p_value.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # Contingency: b_wrong & a_right vs a_wrong & b_right
    b01 = np.sum(correct_a & ~correct_b)  # A right, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B right
    n = b01 + b10
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"chi2": chi2, "p_value": p_value}


def friedman_nemenyi(score_matrix, alpha=0.05):
    """
    Friedman test with Nemenyi post-hoc for multiple model comparison.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        Rows = folds/datasets, columns = models, values = metric scores.

    Returns
    -------
    dict with friedman_stat, friedman_p, avg_ranks, cd (critical difference).
    """
    k = score_matrix.shape[1]  # number of models
    n = score_matrix.shape[0]  # number of groups

    # Ranks per row (1 = best = highest score)
    ranks = score_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean()

    stat, p_val = stats.friedmanchisquare(
        *[score_matrix.iloc[:, i] for i in range(k)]
    )

    # Nemenyi critical difference
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return {
        "friedman_statistic": stat,
        "friedman_p_value": p_val,
        "avg_ranks": avg_ranks.to_dict(),
        "critical_difference": cd,
    }


# ──────────────────────────────────────────────
# Timing utilities
# ──────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def benchmark_model(model, X_train, y_train, X_test, n_repeat=3):
    """
    Measure training time and per-1000-sample inference time.

    Returns
    -------
    dict with train_time_s, predict_time_per_1k_s.
    """
    # Training
    train_times = []
    for _ in range(n_repeat):
        with Timer() as t:
            model.fit(X_train, y_train)
        train_times.append(t.elapsed)

    # Inference
    pred_times = []
    for _ in range(n_repeat):
        with Timer() as t:
            model.predict(X_test)
        n_samples = X_test.shape[0]
        pred_times.append(t.elapsed / n_samples * 1000)

    return {
        "train_time_s": float(np.mean(train_times)),
        "predict_time_per_1k_s": float(np.mean(pred_times)),
    }
