"""
Strategies for handling imbalanced datasets:
  - Random oversampling
  - Random undersampling
  - SMOTE (Synthetic Minority Oversampling Technique)
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.utils.config import (
    RANDOM_SEED,
    STRATEGY_ORIGINAL,
    STRATEGY_OVERSAMPLE,
    STRATEGY_UNDERSAMPLE,
    STRATEGY_SMOTE,
)


def apply_oversampling(X, y):
    """
    Randomly duplicate minority-class samples to match the majority class.
    """
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"[Oversampling] {len(X):,} → {len(X_res):,} samples  "
          f"(fraud ratio: {y_res.mean():.3f})")
    return X_res, y_res


def apply_undersampling(X, y):
    """
    Randomly remove majority-class samples to match the minority class.
    """
    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X, y)
    print(f"[Undersampling] {len(X):,} → {len(X_res):,} samples  "
          f"(fraud ratio: {y_res.mean():.3f})")
    return X_res, y_res


def apply_smote(X, y):
    """
    Generate synthetic minority-class samples using SMOTE.
    """
    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[SMOTE] {len(X):,} → {len(X_res):,} samples  "
          f"(fraud ratio: {y_res.mean():.3f})")
    return X_res, y_res


def get_balanced_datasets(X_train, y_train):
    """
    Create four versions of the training data:
      - original (unchanged)
      - oversampled
      - undersampled
      - smote

    Parameters
    ----------
    X_train, y_train : training features & labels.

    Returns
    -------
    dict[str, tuple[X, y]]
    """
    print(f"\n{'='*50}")
    print("Generating balanced dataset variants …")
    print(f"{'='*50}")

    datasets = {
        STRATEGY_ORIGINAL: (X_train, y_train),
        STRATEGY_OVERSAMPLE: apply_oversampling(X_train, y_train),
        STRATEGY_UNDERSAMPLE: apply_undersampling(X_train, y_train),
        STRATEGY_SMOTE: apply_smote(X_train, y_train),
    }
    return datasets


def describe_balance(y, label: str = ""):
    """Print class distribution information."""
    counts = np.bincount(y.astype(int))
    total = len(y)
    fraud_pct = counts[1] / total * 100 if len(counts) > 1 else 0
    print(f"[{label}]  Total: {total:,}  |  Legit: {counts[0]:,}  |  "
          f"Fraud: {counts[1]:,} ({fraud_pct:.2f}%)")
