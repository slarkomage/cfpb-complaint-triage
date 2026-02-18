"""Metrics helpers for evaluation and plotting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute weighted F1 score."""
    return float(f1_score(y_true, y_pred, average="weighted"))


def confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix array."""
    return confusion_matrix(y_true, y_pred)
