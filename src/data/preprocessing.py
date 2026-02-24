"""
Preprocessing pipelines for European and Sparkov datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EU_DATASET_FILE,
    SPARKOV_DATASET_FILE,
    SPARKOV_TEST_FILE,
    SPARKOV_DROP_COLS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


# ──────────────────────────────────────────────
# European dataset
# ──────────────────────────────────────────────

def load_european_raw(path: Path = None) -> pd.DataFrame:
    """Load the raw European creditcard CSV."""
    path = path or (RAW_DATA_DIR / EU_DATASET_FILE)
    df = pd.read_csv(path)
    print(f"[EU] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def preprocess_european(df: pd.DataFrame = None, save: bool = True):
    """
    Preprocess the European dataset:
      - Check for missing values
      - Scale 'Amount' and 'Time' with StandardScaler
      - 70 / 15 / 15 train / val / test split

    Returns
    -------
    dict with keys X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    if df is None:
        df = load_european_raw()

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"[EU] Missing values: {missing}")
    if missing > 0:
        df = df.dropna()

    # Separate features and target
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    # Scale Amount and Time
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    # Split: first train vs (val+test), then val vs test
    test_val_ratio = VAL_RATIO + TEST_RATIO  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_ratio, random_state=RANDOM_SEED, stratify=y
    )
    relative_val = VAL_RATIO / test_val_ratio  # 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val),
        random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"[EU] Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"[EU] Fraud ratio – train {y_train.mean():.4f}  val {y_val.mean():.4f}  test {y_test.mean():.4f}")

    data = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }

    if save:
        out = PROCESSED_DATA_DIR / "european"
        out.mkdir(parents=True, exist_ok=True)
        for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            data[key].to_csv(out / f"{key}.csv", index=False)
        joblib.dump(scaler, out / "scaler.joblib")
        print(f"[EU] Saved processed data to {out}")

    return data


# ──────────────────────────────────────────────
# Sparkov dataset
# ──────────────────────────────────────────────

def load_sparkov_raw(train_path: Path = None, test_path: Path = None) -> pd.DataFrame:
    """Load and concatenate the Sparkov train & test CSVs."""
    train_path = train_path or (RAW_DATA_DIR / SPARKOV_DATASET_FILE)
    test_path = test_path or (RAW_DATA_DIR / SPARKOV_TEST_FILE)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"[SP] Loaded {len(df):,} rows, {df.shape[1]} columns (train+test).")
    return df


def preprocess_sparkov(df: pd.DataFrame = None, save: bool = True):
    """
    Preprocess the Sparkov dataset:
      - Drop irrelevant columns
      - Convert dates, extract temporal features
      - Label-encode categoricals (category, job)
      - Standardise numerics
      - 70 / 15 / 15 split

    Returns
    -------
    dict with keys X_train, X_val, X_test, y_train, y_val, y_test,
                    scaler, label_encoders, feature_names
    """
    if df is None:
        df = load_sparkov_raw()

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Drop irrelevant columns (only those that exist)
    cols_to_drop = [c for c in SPARKOV_DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # ── Datetime features ──
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"])
        df["trans_year"] = dt.dt.year
        df["trans_month"] = dt.dt.month
        df["trans_day"] = dt.dt.day
        df["trans_hour"] = dt.dt.hour
        df["trans_dayofweek"] = dt.dt.dayofweek
        df = df.drop("trans_date_trans_time", axis=1)

    if "dob" in df.columns:
        dob = pd.to_datetime(df["dob"])
        df["age"] = (pd.Timestamp.now() - dob).dt.days // 365
        df = df.drop("dob", axis=1)

    # ── Encode categoricals ──
    label_encoders = {}
    for col in ["category", "job"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # ── Separate target ──
    y = df["is_fraud"].copy()
    X = df.drop("is_fraud", axis=1).copy()

    # ── Standardise numerics ──
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # ── Split ──
    test_val_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_ratio, random_state=RANDOM_SEED, stratify=y
    )
    relative_val = VAL_RATIO / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val),
        random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"[SP] Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"[SP] Fraud ratio – train {y_train.mean():.4f}  val {y_val.mean():.4f}  test {y_test.mean():.4f}")

    data = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": list(X.columns),
    }

    if save:
        out = PROCESSED_DATA_DIR / "sparkov"
        out.mkdir(parents=True, exist_ok=True)
        for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            data[key].to_csv(out / f"{key}.csv", index=False)
        joblib.dump(scaler, out / "scaler.joblib")
        joblib.dump(label_encoders, out / "label_encoders.joblib")
        print(f"[SP] Saved processed data to {out}")

    return data


# ──────────────────────────────────────────────
# Convenience loader (processed data)
# ──────────────────────────────────────────────

def load_processed(dataset_name: str):
    """
    Load previously processed splits from disk.

    Parameters
    ----------
    dataset_name : str – 'european' or 'sparkov'
    """
    base = PROCESSED_DATA_DIR / dataset_name
    data = {}
    for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = base / f"{key}.csv"
        loaded = pd.read_csv(path)
        if key.startswith("y_"):
            loaded = loaded.squeeze()
        data[key] = loaded
    data["scaler"] = joblib.load(base / "scaler.joblib")
    if dataset_name == "sparkov":
        data["label_encoders"] = joblib.load(base / "label_encoders.joblib")
    data["feature_names"] = list(data["X_train"].columns)
    print(f"[{dataset_name}] Loaded processed data from {base}")
    return data


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds in ("european", "all"):
        preprocess_european()
    if ds in ("sparkov", "all"):
        preprocess_sparkov()
