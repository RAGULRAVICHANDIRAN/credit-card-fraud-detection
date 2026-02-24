"""
Visualization utilities for the Credit Card Fraud Detection project.

All plotting functions save figures to reports/figures/ and return the
matplotlib Figure object for optional inline display in notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; notebooks override this
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config import FIGURES_DIR, ALL_STRATEGIES, ALL_MODELS
from src.utils.metrics import get_roc_curve

# ──────────────────────────────────────────────
# Style defaults
# ──────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

MODEL_COLORS = {
    "naive_bayes": "#4C72B0",
    "random_forest": "#55A868",
    "xgboost": "#C44E52",
    "cnn_bigru": "#8172B2",
    "bert": "#CCB974",
    "stacking": "#64B5CD",
}

STRATEGY_MARKERS = {
    "original": "o",
    "oversampled": "s",
    "undersampled": "D",
    "smote": "^",
}

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _savefig(fig, name, save_dir=None):
    save_dir = Path(save_dir or FIGURES_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved {path}")
    return path


# ══════════════════════════════════════════════
# EDA plots
# ══════════════════════════════════════════════

def plot_class_distribution(y, dataset_name="", save_dir=None):
    """Bar chart of class counts (Legit vs Fraud)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["#55A868", "#C44E52"]
    ax.bar(labels, counts.values, color=colors, edgecolor="black")
    for i, v in enumerate(counts.values):
        ax.text(i, v + v * 0.02, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title(f"Class Distribution – {dataset_name}")
    ax.set_ylabel("Count")
    _savefig(fig, f"class_dist_{dataset_name}", save_dir)
    return fig


def plot_feature_distributions(df, features, target_col="Class",
                               dataset_name="", save_dir=None):
    """KDE plots for selected features split by class."""
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if feat not in df.columns:
            continue
        ax = axes[i]
        for cls, color in zip([0, 1], ["#55A868", "#C44E52"]):
            subset = df[df[target_col] == cls][feat]
            ax.hist(subset, bins=50, alpha=0.5, color=color,
                    label="Legit" if cls == 0 else "Fraud", density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Feature Distributions – {dataset_name}", y=1.01)
    plt.tight_layout()
    _savefig(fig, f"feature_dist_{dataset_name}", save_dir)
    return fig


def plot_correlation_matrix(df, dataset_name="", save_dir=None):
    """Heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, fmt=".1f",
                annot=False)
    ax.set_title(f"Correlation Matrix – {dataset_name}")
    _savefig(fig, f"correlation_{dataset_name}", save_dir)
    return fig


# ══════════════════════════════════════════════
# ROC curves
# ══════════════════════════════════════════════

def plot_roc_curves(roc_data, title="ROC Curves", save_name="roc_curves",
                    save_dir=None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    roc_data : list of dicts, each with keys:
        label, y_true, y_prob
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in roc_data:
        fpr, tpr, _ = get_roc_curve(item["y_true"], item["y_prob"])
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{item["label"]} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _savefig(fig, save_name, save_dir)
    return fig


# ══════════════════════════════════════════════
# Performance comparison bar charts
# ══════════════════════════════════════════════

def plot_performance_by_dataset(results_df, save_dir=None):
    """
    Bar chart comparing model performance metrics across datasets.
    (Figure 3 equivalent)
    """
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(4 * len(METRIC_NAMES), 5))
    for i, metric in enumerate(METRIC_NAMES):
        ax = axes[i]
        pivot = results_df.pivot_table(values=metric, index="dataset",
                                        columns="model", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.suptitle("Classifier Performance by Dataset", y=1.02, fontsize=14)
    plt.tight_layout()
    _savefig(fig, "performance_by_dataset", save_dir)
    return fig


def plot_performance_per_dataset(results_df, dataset_name, save_dir=None):
    """
    Subplot grid: for a single dataset, show performance of all models
    under each balancing strategy.  (Figures 4 & 5 equivalent)
    """
    strategies = results_df["strategy"].unique()
    n_strats = len(strategies)
    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 5))
    if n_strats == 1:
        axes = [axes]

    df_ds = results_df[results_df["dataset"] == dataset_name]

    for ax, strat in zip(axes, strategies):
        df_s = df_ds[df_ds["strategy"] == strat]
        x = np.arange(len(METRIC_NAMES))
        width = 0.8 / max(len(df_s["model"].unique()), 1)
        for j, model in enumerate(df_s["model"].unique()):
            row = df_s[df_s["model"] == model].iloc[0]
            vals = [row.get(m, 0) for m in METRIC_NAMES]
            ax.bar(x + j * width, vals, width,
                   label=model, color=MODEL_COLORS.get(model, None),
                   edgecolor="black")
        ax.set_xticks(x + width)
        ax.set_xticklabels(METRIC_NAMES, rotation=45)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{strat.title()}")
        ax.legend(fontsize=7)

    fig.suptitle(f"Performance on {dataset_name.title()} Dataset",
                 y=1.02, fontsize=14)
    plt.tight_layout()
    _savefig(fig, f"performance_{dataset_name}", save_dir)
    return fig


# ══════════════════════════════════════════════
# Density / distribution of F1 scores
# ══════════════════════════════════════════════

def plot_f1_density(results_df, save_dir=None):
    """
    Density probability distributions of F1 score:
      (a) by dataset   (b) by model    (Figure 6 equivalent)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By dataset
    for ds in results_df["dataset"].unique():
        subset = results_df[results_df["dataset"] == ds]["f1"].dropna()
        if len(subset) > 1:
            axes[0].hist(subset, bins=10, alpha=0.5, label=ds, density=True)
    axes[0].set_title("F1 Distribution by Dataset")
    axes[0].set_xlabel("F1 Score")
    axes[0].legend()

    # By model
    for model in results_df["model"].unique():
        subset = results_df[results_df["model"] == model]["f1"].dropna()
        if len(subset) > 1:
            axes[1].hist(subset, bins=10, alpha=0.5, label=model, density=True)
    axes[1].set_title("F1 Distribution by Model")
    axes[1].set_xlabel("F1 Score")
    axes[1].legend()

    plt.tight_layout()
    _savefig(fig, "f1_density", save_dir)
    return fig


# ══════════════════════════════════════════════
# Average performance comparison
# ══════════════════════════════════════════════

def plot_average_performance(results_df, save_dir=None):
    """
    Figure 7 equivalent:
      (a) Average F1 of each classifier on both datasets.
      (b) Gap in performance between EU and Sparkov.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Average F1
    avg = results_df.groupby(["model", "dataset"])["f1"].mean().unstack()
    avg.plot(kind="bar", ax=axes[0], edgecolor="black")
    axes[0].set_title("Average F1 by Classifier & Dataset")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].legend(title="Dataset")

    # (b) Gap
    if "european" in avg.columns and "sparkov" in avg.columns:
        gap = avg["european"] - avg["sparkov"]
        gap.plot(kind="bar", ax=axes[1], color="#8172B2", edgecolor="black")
        axes[1].set_title("F1 Gap (European − Sparkov)")
        axes[1].set_ylabel("F1 Difference")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    _savefig(fig, "average_performance", save_dir)
    return fig


# ══════════════════════════════════════════════
# Confusion matrix
# ══════════════════════════════════════════════

def plot_confusion_matrix(cm, title="Confusion Matrix", save_name="cm",
                          save_dir=None):
    """Plot a single confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _savefig(fig, save_name, save_dir)
    return fig


def plot_confusion_matrices_grid(cm_dict, save_dir=None):
    """
    Plot a grid of confusion matrices.

    Parameters
    ----------
    cm_dict : dict  – {label: confusion_matrix_array}
    """
    n = len(cm_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (label, cm) in enumerate(cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["L", "F"], yticklabels=["L", "F"],
                    ax=axes[i])
        axes[i].set_title(label, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices", y=1.01)
    plt.tight_layout()
    _savefig(fig, "confusion_matrices_grid", save_dir)
    return fig


# ══════════════════════════════════════════════
# Training history (deep learning)
# ══════════════════════════════════════════════

def plot_training_history(history, model_name="Model", save_dir=None):
    """Plot Keras training history (loss, accuracy, AUC)."""
    h = history.history if hasattr(history, "history") else history
    metrics_to_plot = [k for k in h.keys() if not k.startswith("val_")]
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics_to_plot):
        ax.plot(h[metric], label=f"Train {metric}")
        if f"val_{metric}" in h:
            ax.plot(h[f"val_{metric}"], label=f"Val {metric}")
        ax.set_title(metric.title())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(f"{model_name} Training History", y=1.02)
    plt.tight_layout()
    _savefig(fig, f"training_history_{model_name.lower().replace(' ', '_')}",
             save_dir)
    return fig


# ══════════════════════════════════════════════
# Time complexity comparison
# ══════════════════════════════════════════════

def plot_time_comparison(timing_df, save_dir=None):
    """
    Bar chart of training time and prediction time per 1 k samples.

    Parameters
    ----------
    timing_df : pd.DataFrame with columns:
        model, train_time_s, predict_time_per_1k_s
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    timing_df.set_index("model")[["train_time_s"]].plot(
        kind="bar", ax=axes[0], color="#4C72B0", edgecolor="black", legend=False
    )
    axes[0].set_title("Training Time (seconds)")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    timing_df.set_index("model")[["predict_time_per_1k_s"]].plot(
        kind="bar", ax=axes[1], color="#C44E52", edgecolor="black", legend=False
    )
    axes[1].set_title("Prediction Time per 1 k Samples")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    _savefig(fig, "time_comparison", save_dir)
    return fig


# ══════════════════════════════════════════════
# Research methodology flowchart (Figure 1)
# ══════════════════════════════════════════════

def plot_methodology_flowchart(save_dir=None):
    """
    Create a simplified research methodology flowchart (Figure 1).
    Uses matplotlib patches & arrows.
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#D6EAF8",
                     edgecolor="#2C3E50", linewidth=1.5)
    arrow_props = dict(arrowstyle="->", color="#2C3E50", lw=2)

    steps = [
        (5, 18.5, "Data Collection\n(European & Sparkov Datasets)"),
        (5, 16.5, "Data Preprocessing\n(Scaling, Encoding, Feature Extraction)"),
        (5, 14.5, "Handling Imbalanced Data\n(Oversampling / Undersampling / SMOTE)"),
        (5, 12.5, "Train-Validation-Test Split\n(70% / 15% / 15%)"),
        (3, 10.5, "Existing Models\n(NB, RF, XGBoost)"),
        (7, 10.5, "Proposed Models\n(CNN-BiGRU, BERT, Stacking)"),
        (5, 8.5,  "Model Evaluation\n(Accuracy, Precision, Recall, F1, AUC)"),
        (5, 6.5,  "Explainability Analysis\n(SHAP, LIME, Feature Importance)"),
        (5, 4.5,  "Comparative Analysis &\nStatistical Significance Tests"),
        (5, 2.5,  "Results & Conclusions"),
    ]

    for x, y, text in steps:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                bbox=box_style, fontweight="bold")

    # Arrows between sequential steps
    seq = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6),
           (6, 7), (7, 8), (8, 9)]
    for a, b in seq:
        ax.annotate("", xy=(steps[b][0], steps[b][1] + 0.6),
                    xytext=(steps[a][0], steps[a][1] - 0.6),
                    arrowprops=arrow_props)

    fig.suptitle("Research Methodology Flowchart", fontsize=14,
                 fontweight="bold", y=0.98)
    _savefig(fig, "methodology_flowchart", save_dir)
    return fig
