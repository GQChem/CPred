"""Artificial Neural Network model for CP site prediction.

3-layer ANN: Linear(n_features, hidden) -> Sigmoid -> Linear(hidden, 1) -> Sigmoid
where hidden = round(sqrt(n_features * 1)) = round(sqrt(n_features))

Trained with BCELoss, SGD optimizer (lr=0.5, momentum=0.1), 5000 iterations
each picking one random sample (Lo et al. 2012).

Multiple restarts (default 5) with different random seeds; the model with
best training loss is kept.

Falls back to a simple sklearn MLPClassifier if PyTorch is not available.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


if HAS_TORCH:

    class CPredANNModule(nn.Module):
        """PyTorch ANN module: F -> round(sqrt(F)) -> 1 with sigmoid activations."""

        def __init__(self, n_features: int = 46):
            super().__init__()
            # Hidden neurons: Round(sqrt(N_input * N_output)) = Round(sqrt(n_features * 1))
            hidden = max(round(math.sqrt(n_features * 1)), 2)
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.Sigmoid(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            # Initialize weights in [-2, +2] per paper (page 16)
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -2.0, 2.0)
                    nn.init.uniform_(layer.bias, -2.0, 2.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class CPredANN:
    """ANN classifier for CP site prediction."""

    def __init__(self, n_features: int = 46, lr: float = 0.5,
                 momentum: float = 0.1, n_iterations: int = 5000,
                 n_restarts: int = 5):
        self.n_features = n_features
        self.lr = lr
        self.momentum = momentum
        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self._fitted = False
        self._use_torch = HAS_TORCH
        self._model = None
        self._sklearn_model = None

        if HAS_TORCH:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self._model = CPredANNModule(n_features).to(self.device)

    def _train_one(self, X_t: 'torch.Tensor', y_t: 'torch.Tensor',
                   seed: int) -> tuple['CPredANNModule', float]:
        """Train a single ANN with given seed. Returns (model, final_loss)."""
        torch.manual_seed(seed)
        model = CPredANNModule(self.n_features).to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=self.momentum)

        n_samples = len(y_t)
        rng = np.random.RandomState(seed)

        model.train()
        for _ in range(self.n_iterations):
            idx = rng.randint(0, n_samples)
            optimizer.zero_grad()
            pred = model(X_t[idx:idx+1])
            loss = nn.functional.binary_cross_entropy(pred, y_t[idx:idx+1])
            loss.backward()
            optimizer.step()

        # Compute full training loss for model selection
        model.eval()
        with torch.no_grad():
            all_pred = model(X_t)
            full_loss = nn.functional.binary_cross_entropy(all_pred, y_t).item()

        return model, full_loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ANN with multiple restarts, keep best."""
        self.n_features = X.shape[1]

        if self._use_torch:
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)

            best_model = None
            best_loss = float('inf')

            for restart in range(self.n_restarts):
                model, loss = self._train_one(X_t, y_t, seed=42 + restart)
                if loss < best_loss:
                    best_loss = loss
                    best_model = model

            self._model = best_model
        else:
            from sklearn.neural_network import MLPClassifier
            hidden = max(round(math.sqrt(self.n_features)), 2)
            self._sklearn_model = MLPClassifier(
                hidden_layer_sizes=(hidden,),
                activation="logistic",
                max_iter=self.n_iterations,
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
