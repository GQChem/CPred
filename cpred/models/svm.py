"""SVM model for CP site prediction.

RBF kernel with probability calibration, as specified in Lo et al. (2012).
Features are already Z-score normalized, so no StandardScaler is needed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class CPredSVM:
    """SVM classifier for CP site prediction."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale"):
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            grid_search: bool = True) -> None:
        """Train the SVM model, optionally with grid search."""
        if grid_search:
            param_grid = {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "gamma": ["scale", "auto", 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            }
            gs = GridSearchCV(
                self.model, param_grid,
                scoring="roc_auc", cv=5, n_jobs=-1, verbose=0,
            )
            gs.fit(X, y)
            self.model = gs.best_estimator_
        else:
            self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._fitted = True
