"""Training routines for CPred models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cpred.models.ensemble import CPredEnsemble
from cpred.pipeline import FEATURE_NAMES
from cpred.training.evaluate import evaluate_model


def train_ensemble(X: np.ndarray, y: np.ndarray,
                   feature_names: list[str] | None = None,
                   output_dir: Path | None = None) -> CPredEnsemble:
    """Train the full CPred ensemble model.

    Args:
        X: (N, F) feature matrix.
        y: (N,) binary labels.
        feature_names: Feature names for HI model.
        output_dir: Directory to save trained models.

    Returns:
        Trained CPredEnsemble.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    print(f"Training on {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive samples: {int(y.sum())} ({y.mean() * 100:.1f}%)")

    ensemble = CPredEnsemble(feature_names=feature_names)
    ensemble.fit(X, y, feature_names=feature_names)

    # Evaluate on training data
    print("\n--- Training set performance ---")
    metrics = evaluate_model(ensemble, X, y)
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ensemble.save(output_dir)
        print(f"\nModels saved to {output_dir}")

    return ensemble
