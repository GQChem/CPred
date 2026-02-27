"""SVM model for CP site prediction.

RBF kernel with probability calibration, as specified in Lo et al. (2012).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class CPredSVM:
    """SVM classifier for CP site prediction."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale"):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42,
            )),
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            grid_search: bool = True) -> None:
        """Train the SVM model, optionally with grid search.

        Args:
            X: (N, F) feature matrix.
            y: (N,) binary labels.
            grid_search: Whether to perform grid search for C and gamma.
        """
        if grid_search:
            param_grid = {
                "svm__C": [0.1, 1.0, 10.0, 100.0],
                "svm__gamma": ["scale", "auto", 0.01, 0.1],
            }
            gs = GridSearchCV(
                self.pipeline, param_grid,
                scoring="roc_auc", cv=5, n_jobs=-1, verbose=0,
            )
            gs.fit(X, y)
            self.pipeline = gs.best_estimator_
        else:
            self.pipeline.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities."""
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self._fitted = True
