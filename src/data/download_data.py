"""
Dataset download helper for the European Cardholders dataset.
"""

from pathlib import Path
from src.utils.config import RAW_DATA_DIR, DATASET_FILE, DATASET_URL


def download_dataset():
    """Download the European dataset via Kaggle API or print instructions."""
    dest = RAW_DATA_DIR / DATASET_FILE
    if dest.exists():
        print(f"[✓] Dataset already exists: {dest}")
        return

    print("📦 Checking / downloading dataset …")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(RAW_DATA_DIR),
            unzip=True,
        )
        print(f"[✓] Downloaded to {RAW_DATA_DIR}")
    except Exception:
        print(
            f"[!] Could not auto-download. Please download manually:\n"
            f"    URL:  {DATASET_URL}\n"
            f"    Save: {dest}\n"
            f"    Or run: kaggle datasets download mlg-ulb/creditcardfraud -p {RAW_DATA_DIR} --unzip"
        )
