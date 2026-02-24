"""
Dataset download helpers and instructions.

The Kaggle datasets require authentication.  This module:
  1. Checks if the raw CSV files already exist.
  2. Tries to download via the Kaggle CLI (if `kaggle` is installed and authenticated).
  3. Falls back to printing clear manual-download instructions.
"""

import os
import subprocess
from pathlib import Path

from src.utils.config import (
    RAW_DATA_DIR,
    EU_DATASET_FILE,
    SPARKOV_DATASET_FILE,
    SPARKOV_TEST_FILE,
    EU_DATASET_URL,
    SPARKOV_DATASET_URL,
)


def _kaggle_available() -> bool:
    """Return True if the kaggle CLI is installed and a token exists."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _download_kaggle_dataset(dataset_slug: str, dest: Path):
    """Download a Kaggle dataset using the CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug,
         "-p", str(dest), "--unzip"],
        check=True,
    )


def download_european_dataset(force: bool = False):
    """
    Download the European Cardholders (creditcard.csv) dataset.
    """
    target = RAW_DATA_DIR / EU_DATASET_FILE
    if target.exists() and not force:
        print(f"[✓] European dataset already exists: {target}")
        return target

    if _kaggle_available():
        print("[↓] Downloading European dataset via Kaggle API …")
        _download_kaggle_dataset("mlg-ulb/creditcardfraud", RAW_DATA_DIR)
        if target.exists():
            print(f"[✓] Downloaded: {target}")
            return target

    # Manual instructions
    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED – European Cardholders Dataset")
    print("=" * 70)
    print(f"1. Visit  {EU_DATASET_URL}")
    print("2. Sign in to Kaggle and click 'Download'.")
    print(f"3. Extract the CSV and place it at:\n   {target}")
    print("=" * 70)
    return None


def download_sparkov_dataset(force: bool = False):
    """
    Download the Sparkov simulated dataset (fraudTrain.csv / fraudTest.csv).
    """
    target_train = RAW_DATA_DIR / SPARKOV_DATASET_FILE
    target_test = RAW_DATA_DIR / SPARKOV_TEST_FILE
    if target_train.exists() and target_test.exists() and not force:
        print(f"[✓] Sparkov dataset already exists: {RAW_DATA_DIR}")
        return target_train, target_test

    if _kaggle_available():
        print("[↓] Downloading Sparkov dataset via Kaggle API …")
        _download_kaggle_dataset("kartik2112/fraud-detection", RAW_DATA_DIR)
        if target_train.exists():
            print(f"[✓] Downloaded: {target_train}")
            return target_train, target_test

    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED – Sparkov Simulated Dataset")
    print("=" * 70)
    print(f"1. Visit  {SPARKOV_DATASET_URL}")
    print("2. Sign in to Kaggle and click 'Download'.")
    print(f"3. Extract the CSVs and place them at:\n   {target_train}\n   {target_test}")
    print("=" * 70)
    return None


def download_all(force: bool = False):
    """Attempt to download all required datasets."""
    print("\n📦 Checking / downloading datasets …\n")
    eu = download_european_dataset(force=force)
    sp = download_sparkov_dataset(force=force)
    return {"european": eu, "sparkov": sp}


if __name__ == "__main__":
    download_all()
