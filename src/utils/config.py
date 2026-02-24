"""
Project-wide configuration: paths, constants, hyperparameters, and random seeds.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
           MODELS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset information (European Cardholders only)
# ──────────────────────────────────────────────
DATASET_NAME = "european"
DATASET_FILE = "creditcard.csv"
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
TARGET_COL = "Class"

# ──────────────────────────────────────────────
# Train / Validation / Test split ratios
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
N_SPLITS = 3

# ──────────────────────────────────────────────
# Model hyper-parameters  (paper defaults)
# ──────────────────────────────────────────────
PAPER_RF_PARAMS = {
    "n_estimators": 5,
    "max_samples": 0.2,
    "max_features": 0.3,
    "max_depth": 3,
    "random_state": RANDOM_SEED,
}

PAPER_XGB_PARAMS = {
    "n_estimators": 5,
    "max_depth": 3,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

# ──────────────────────────────────────────────
# Hyperparameter search grids (proposed work)
# ──────────────────────────────────────────────
TUNE_RF_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
}

TUNE_XGB_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
}

# ──────────────────────────────────────────────
# Deep-learning hyper-parameters
# ──────────────────────────────────────────────
CNN_BIGRU_CONFIG = {
    "conv_filters": 64,
    "kernel_size": 3,
    "gru_units": 64,
    "dropout_rate": 0.5,
    "dense_units": 32,
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
}

BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 2e-5,
}

# ──────────────────────────────────────────────
# Balancing strategy names
# ──────────────────────────────────────────────
STRATEGY_ORIGINAL = "original"
STRATEGY_OVERSAMPLE = "oversampled"
STRATEGY_UNDERSAMPLE = "undersampled"
STRATEGY_SMOTE = "smote"
ALL_STRATEGIES = [STRATEGY_ORIGINAL, STRATEGY_OVERSAMPLE,
                  STRATEGY_UNDERSAMPLE, STRATEGY_SMOTE]

# ──────────────────────────────────────────────
# Model registry names
# ──────────────────────────────────────────────
MODEL_NB = "naive_bayes"
MODEL_RF = "random_forest"
MODEL_XGB = "xgboost"
MODEL_CNN_BIGRU = "cnn_bigru"
MODEL_BERT = "bert"
MODEL_STACKING = "stacking"
EXISTING_MODELS = [MODEL_NB, MODEL_RF, MODEL_XGB]
PROPOSED_MODELS = [MODEL_CNN_BIGRU, MODEL_BERT, MODEL_STACKING]
ALL_MODELS = EXISTING_MODELS + PROPOSED_MODELS
