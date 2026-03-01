"""Random Forest model for CP site prediction.

Per Lo et al. 2012 (page 17):
- Grow 1,000 trees, each with max_features = 0.5 * n_features (C4.5 style)
- Bootstrap sample size = n_training
- Sort trees by MCC on their out-of-bag samples, keep top 500
- Final prediction = proportion of top-500 trees predicting viable
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef


class CPredRandomForest:
    """Random Forest classifier for CP site prediction.

    Grows 1000 trees, selects top 500 by OOB MCC, per Lo et al. 2012.
    """

    def __init__(self, n_estimators_grow: int = 1000,
                 n_estimators_keep: int = 500,
                 random_state: int = 42):
        self.n_estimators_grow = n_estimators_grow
        self.n_estimators_keep = n_estimators_keep
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Grow 1000 trees, keep top 500 by OOB MCC."""
        n_samples, n_features = X.shape
        max_features = max(1, round(0.5 * n_features))
        rng = np.random.RandomState(self.random_state)

        tree_mccs: list[tuple[float, DecisionTreeClassifier]] = []

        for i in range(self.n_estimators_grow):
            seed = int(rng.randint(0, 2**31))
            # Bootstrap sample
            idx_boot = rng.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(idx_boot)] = False
            X_boot, y_boot = X[idx_boot], y[idx_boot]

            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_features=max_features,
                random_state=seed,
            )
            tree.fit(X_boot, y_boot)

            # Score on OOB samples
            if oob_mask.sum() >= 2 and len(np.unique(y[oob_mask])) == 2:
                oob_pred = tree.predict(X[oob_mask])
                mcc = matthews_corrcoef(y[oob_mask], oob_pred)
            else:
                mcc = 0.0

            tree_mccs.append((mcc, tree))

        # Keep top 500 trees by MCC
        tree_mccs.sort(key=lambda x: x[0], reverse=True)
        self.trees_ = [t for _, t in tree_mccs[:self.n_estimators_keep]]
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict as proportion of top-500 trees voting viable."""
        votes = np.array([t.predict(X) for t in self.trees_])  # (n_trees, n_samples)
        return votes.mean(axis=0)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.trees_, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.trees_ = pickle.load(f)
        self._fitted = True
