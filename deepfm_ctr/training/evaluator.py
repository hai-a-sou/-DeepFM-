"""Evaluation metrics for CTR prediction."""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_auc(labels, scores):
    """Compute ROC-AUC."""
    return roc_auc_score(labels, scores)


def evaluate_logloss(labels, scores):
    """Compute binary cross-entropy (log loss)."""
    eps = 1e-7
    scores = np.clip(scores, eps, 1 - eps)
    return -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))


def evaluate_all(labels, scores, threshold=0.5):
    """Compute AUC, log loss, and classification metrics."""
    auc = evaluate_auc(labels, scores)
    logloss = evaluate_logloss(labels, scores)

    preds = (scores >= threshold).astype(np.int32)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = np.mean(preds == labels)

    return {
        "auc": auc,
        "logloss": logloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
