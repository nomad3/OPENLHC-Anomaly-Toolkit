from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class BaselinePCA:
    n_components: int = 4
    pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> None:
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "PCA not fitted"
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        err = ((X - X_rec) ** 2).sum(axis=1)
        return err.astype(np.float32)


@dataclass
class BaselineIForest:
    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: float = 0.01
    model: Optional[IsolationForest] = None

    def fit(self, X: np.ndarray) -> None:
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=42,
        )
        self.model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "IsolationForest not fitted"
        # Higher means more anomalous: use negative of score_samples
        s = -self.model.score_samples(X)
        return s.astype(np.float32)
