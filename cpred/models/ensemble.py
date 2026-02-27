"""Ensemble model: simple average of ANN, SVM, RF, and HI predictions.

Final probability = (P_ann + P_svm + P_rf + P_hi) / 4
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cpred.models.ann import CPredANN
from cpred.models.svm import CPredSVM
from cpred.models.random_forest import CPredRandomForest
from cpred.models.hierarchical import CPredHierarchical


class CPredEnsemble:
    """Ensemble of ANN + SVM + RF + HI models."""

    def __init__(self, feature_names: list[str] | None = None):
        self.ann = CPredANN()
        self.svm = CPredSVM()
        self.rf = CPredRandomForest()
        self.hi = CPredHierarchical(feature_names=feature_names or [])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> None:
        """Train all four component models.

        Args:
            X: (N, F) feature matrix.
            y: (N,) binary labels.
            feature_names: Names of features for HI model.
        """
        print("Training Random Forest...")
        self.rf.fit(X, y)

        print("Training SVM...")
        self.svm.fit(X, y, grid_search=True)

        print("Training ANN...")
        self.ann.fit(X, y)

        print("Training Hierarchical model...")
        self.hi.fit(X, y, feature_names=feature_names)

        self._fitted = True
        print("Ensemble training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability as average of 4 model probabilities.

        Returns:
            (N,) probability scores in [0, 1].
        """
        p_rf = self.rf.predict(X)
        p_svm = self.svm.predict(X)
        p_ann = self.ann.predict(X)
        p_hi = self.hi.predict(X)

        return (p_rf + p_svm + p_ann + p_hi) / 4.0

    def predict_individual(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict from each model individually.

        Returns:
            Dict mapping model name to (N,) probability arrays.
        """
        return {
            "rf": self.rf.predict(X),
            "svm": self.svm.predict(X),
            "ann": self.ann.predict(X),
            "hi": self.hi.predict(X),
        }

    def save(self, model_dir: str | Path) -> None:
        """Save all models to a directory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.rf.save(model_dir / "rf.pkl")
        self.svm.save(model_dir / "svm.pkl")
        self.ann.save(model_dir / "ann.pt")
        self.hi.save(model_dir / "hi.json")

    def load(self, model_dir: str | Path) -> None:
        """Load all models from a directory."""
        model_dir = Path(model_dir)

        rf_path = model_dir / "rf.pkl"
        svm_path = model_dir / "svm.pkl"
        ann_path = model_dir / "ann.pt"
        hi_path = model_dir / "hi.json"

        if rf_path.exists():
            self.rf.load(rf_path)
        if svm_path.exists():
            self.svm.load(svm_path)
        if ann_path.exists():
            self.ann.load(ann_path)
        if hi_path.exists():
            self.hi.load(hi_path)

        self._fitted = True
