"""Artificial Neural Network model for CP site prediction.

3-layer ANN: Linear(46, 23) -> Sigmoid -> Linear(23, 1) -> Sigmoid
Trained with BCELoss and Adam optimizer.

Falls back to a simple sklearn MLPClassifier if PyTorch is not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


if HAS_TORCH:

    class CPredANNModule(nn.Module):
        """PyTorch ANN module: F -> F//2 -> 1 with sigmoid activations."""

        def __init__(self, n_features: int = 19):
            super().__init__()
            hidden = max(n_features // 2, 10)
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.Sigmoid(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class CPredANN:
    """ANN classifier for CP site prediction."""

    def __init__(self, n_features: int = 19, lr: float = 0.001,
                 epochs: int = 100, batch_size: int = 64):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._fitted = False
        self._use_torch = HAS_TORCH
        self._model = None
        self._sklearn_model = None

        if HAS_TORCH:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self._model = CPredANNModule(n_features).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ANN model."""
        self.n_features = X.shape[1]

        # Compute class weights for imbalanced data
        n_pos = max(y.sum(), 1)
        n_neg = max(len(y) - n_pos, 1)
        pos_weight = n_neg / n_pos

        if self._use_torch:
            self._model = CPredANNModule(self.n_features).to(self.device)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

            # Use per-sample weights to handle class imbalance
            sample_weights = torch.where(
                torch.FloatTensor(y) > 0.5,
                torch.tensor(pos_weight, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            ).to(self.device)

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(X_t, y_t,
                                    sample_weights)
            loader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True)

            self._model.train()
            for _ in range(self.epochs):
                for batch_X, batch_y, batch_w in loader:
                    optimizer.zero_grad()
                    pred = self._model(batch_X)
                    # Weighted BCE loss
                    loss = nn.functional.binary_cross_entropy(
                        pred, batch_y, weight=batch_w)
                    loss.backward()
                    optimizer.step()
        else:
            # Fallback to sklearn MLP
            from sklearn.neural_network import MLPClassifier
            hidden = max(self.n_features // 2, 10)
            self._sklearn_model = MLPClassifier(
                hidden_layer_sizes=(hidden,),
                activation="logistic",
                max_iter=self.epochs,
                random_state=42,
            )
            self._sklearn_model.fit(X, y)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities."""
        if self._use_torch:
            self._model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                probs = self._model(X_t).cpu().numpy()
            return probs
        else:
            return self._sklearn_model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        if self._use_torch:
            torch.save({
                "model_state": self._model.state_dict(),
                "n_features": self.n_features,
            }, path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self._sklearn_model, f)

    def load(self, path: str | Path) -> None:
        if self._use_torch:
            checkpoint = torch.load(path, map_location=self.device,
                                    weights_only=True)
            self.n_features = checkpoint["n_features"]
            self._model = CPredANNModule(self.n_features).to(self.device)
            self._model.load_state_dict(checkpoint["model_state"])
        else:
            import pickle
            with open(path, "rb") as f:
                self._sklearn_model = pickle.load(f)
        self._fitted = True
