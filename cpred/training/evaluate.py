"""Evaluation metrics for CPred models.

Includes 10-fold cross-validation, AUC, sensitivity, specificity, and MCC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: (N,) true binary labels.
        y_prob: (N,) predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dict with AUC, sensitivity, specificity, MCC, accuracy.
    """
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "mcc": mcc,
    }


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a trained model on a dataset."""
    probs = model.predict(X)
    return compute_metrics(y, probs, threshold)


def cross_validate(model_class, X: np.ndarray, y: np.ndarray,
                   n_folds: int = 10,
                   feature_names: list[str] | None = None,
                   **model_kwargs) -> dict[str, float]:
    """Perform stratified k-fold cross-validation.

    Args:
        model_class: Class with fit(X, y) and predict(X) methods.
        X: (N, F) feature matrix.
        y: (N,) binary labels.
        n_folds: Number of CV folds.
        feature_names: Feature names (for HI model).
        **model_kwargs: Additional arguments for model constructor.

    Returns:
        Mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**model_kwargs)
        if hasattr(model, "feature_names") and feature_names:
            model.feature_names = feature_names
        model.fit(X_train, y_train)

        probs = model.predict(X_test)
        metrics = compute_metrics(y_test, probs)
        all_metrics.append(metrics)
        print(f"  Fold {fold + 1}/{n_folds}: AUC={metrics['auc']:.4f}")

    # Average across folds
    mean_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        mean_metrics[key] = np.mean(values)
        mean_metrics[f"{key}_std"] = np.std(values)

    return mean_metrics
