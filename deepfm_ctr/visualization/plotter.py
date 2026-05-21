"""Visualization functions for model evaluation."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def plot_roc_curves(labels_dict, scores_dict, save_path):
    """Overlay ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"DeepFM": "#2196F3", "FM": "#FF9800", "XGBoost": "#4CAF50"}

    for name, scores in scores_dict.items():
        labels = labels_dict[name]
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, None), lw=2,
                label=f"{name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Criteo CTR Prediction")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path):
    """Plot training and validation loss + AUC over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, val_loss, "r-s", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_auc, "g-D", markersize=4, label="Val AUC")
    best_epoch = epochs[np.argmax(val_auc)]
    best_auc = max(val_auc)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(f"Best: {best_auc:.4f} @ epoch {best_epoch}",
                 xy=(best_epoch, best_auc), xytext=(best_epoch + 1, best_auc - 0.002),
                 fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("Validation AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_search(dim_results, save_path):
    """Bar/line chart: embedding dimension vs validation AUC."""
    dims = sorted(dim_results.keys())
    aucs = [dim_results[d] for d in dims]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(dims)), aucs, color="#2196F3", alpha=0.8, width=0.6)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Embedding Dimension Search — DeepFM")

    # Label each bar
    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{auc_val:.4f}", ha="center", va="bottom", fontsize=10)

    # Highlight best
    best_idx = np.argmax(aucs)
    bars[best_idx].set_color("#FF5722")

    ax.set_ylim(min(aucs) - 0.002, max(aucs) + 0.003)
    ax.grid(True, alpha=0.3, axis="y")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics, save_path):
    """Grouped bar chart comparing AUC and LogLoss across models."""
    models = list(metrics.keys())
    aucs = [metrics[m]["auc"] for m in models]
    loglosses = [metrics[m]["logloss"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    ax1.bar(models, aucs, color=colors[:len(models)], alpha=0.85)
    ax1.set_ylabel("AUC")
    ax1.set_title("Test AUC by Model")
    ax1.set_ylim(min(aucs) - 0.01, max(aucs) + 0.005)
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(models, loglosses, color=colors[:len(models)], alpha=0.85)
    ax2.set_ylabel("Log Loss")
    ax2.set_title("Test Log Loss by Model")
    ax2.set_ylim(min(loglosses) - 0.005, max(loglosses) + 0.005)
    for i, v in enumerate(loglosses):
        ax2.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_distribution(labels, scores, model_name, save_path):
    """Histogram of predicted probabilities split by true label."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(neg_scores, bins=50, alpha=0.6, label="Negative (no click)", color="#4CAF50", density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, label="Positive (click)", color="#FF5722", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Prediction Distribution — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
