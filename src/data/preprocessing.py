"""
Data preprocessing for the European Cardholders dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_FILE, DATASET_NAME,
    TARGET_COL, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)


def preprocess(df: pd.DataFrame = None, save: bool = True):
    """
    Preprocess the European Cardholders dataset.
    - Scale 'Amount' and 'Time' with StandardScaler.
    - Split into train / validation / test (70/15/15).
    """
    if df is None:
        csv_path = RAW_DATA_DIR / DATASET_FILE
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. "
                f"Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )
        df = pd.read_csv(csv_path)

    print(f"[{DATASET_NAME}] Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Scale Amount and Time
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    # Split
    val_test = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test, random_state=RANDOM_SEED, stratify=y,
    )
    relative_test = TEST_RATIO / val_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=RANDOM_SEED, stratify=y_temp,
    )

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")

    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": list(X.columns),
    }

    if save:
        out_dir = PROCESSED_DATA_DIR / DATASET_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(out_dir / "X_train.csv", index=False)
        y_train.to_csv(out_dir / "y_train.csv", index=False)
        X_val.to_csv(out_dir / "X_val.csv", index=False)
        y_val.to_csv(out_dir / "y_val.csv", index=False)
        X_test.to_csv(out_dir / "X_test.csv", index=False)
        y_test.to_csv(out_dir / "y_test.csv", index=False)
        joblib.dump(scaler, out_dir / "scaler.joblib")
        print(f"  Saved to {out_dir}")

    return data


def load_processed():
    """Load previously preprocessed data."""
    d = PROCESSED_DATA_DIR / DATASET_NAME
    data = {
        "X_train": pd.read_csv(d / "X_train.csv"),
        "y_train": pd.read_csv(d / "y_train.csv").squeeze(),
        "X_val": pd.read_csv(d / "X_val.csv"),
        "y_val": pd.read_csv(d / "y_val.csv").squeeze(),
        "X_test": pd.read_csv(d / "X_test.csv"),
        "y_test": pd.read_csv(d / "y_test.csv").squeeze(),
    }
    print(f"[{DATASET_NAME}] Loaded processed data from {d}")
    return data
