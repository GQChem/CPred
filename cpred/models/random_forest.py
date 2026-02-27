"""Random Forest model for CP site prediction.

500 trees, entropy criterion, as specified in Lo et al. (2012).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CPredRandomForest:
    """Random Forest classifier for CP site prediction."""

    def __init__(self, n_estimators: int = 500, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="entropy",
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model.

        Args:
            X: (N, 46) feature matrix.
            y: (N,) binary labels (1 = viable CP site).
        """
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities.

        Returns:
            (N,) probability of being a viable CP site.
        """
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str | Path) -> None:
        """Load model from pickle file."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._fitted = True
