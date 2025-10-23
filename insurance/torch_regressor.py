from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        hidden: int = 64,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self._model = None
        self._in_dim = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        self._in_dim = X.shape[1]
        model = _MLP(self._in_dim, hidden=self.hidden, dropout=self.dropout).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self._model = model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self._model(X_t).cpu().numpy()
        return pred

