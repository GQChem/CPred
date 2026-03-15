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
            # LIBSVM grid.py defaults: C in 2^{-5,-3,...,15}, gamma in 2^{-15,-13,...,3}
            # (Lo et al. 2012, page 16-17: "determined by the program grid.py
            # (with default settings) included in LIBSVM")
            param_grid = {
                "C": [2**i for i in range(-5, 16, 2)],
                "gamma": [2**i for i in range(-15, 4, 2)],
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
