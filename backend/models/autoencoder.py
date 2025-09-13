from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for the autoencoder model.")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        Act = acts.get(activation, nn.ReLU)
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        mid = hidden_dims[-1] if hidden_dims else input_dim // 2
        enc_dims = hidden_dims
        dec_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        self.encoder = MLP(input_dim, enc_dims, activation, dropout)
        self.decoder = MLP(mid, dec_dims, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


@dataclass
class AEParams:
    hidden_dims: List[int]
    activation: str = "relu"
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    device: str = "cpu"


class AutoencoderTrainer:
    def __init__(self, input_dim: int, params: AEParams):
        self.params = params
        self.device = torch.device(params.device)
        self.model = Autoencoder(input_dim, params.hidden_dims, params.activation, params.dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=params.lr)

    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None) -> None:
        ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.params.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.params.epochs):
            for (xb,) in dl:
                xb = xb.to(self.device)
                self.opt.zero_grad()
                xr = self.model(xb)
                loss = self.criterion(xr, xb)
                loss.backward()
                self.opt.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            xr = self.model(x)
            err = ((x - xr) ** 2).sum(dim=1).cpu().numpy()
        return err.astype(np.float32)

    def save(self, path: Path) -> None:
        from pathlib import Path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "params": self.params.__dict__,
        }, str(path))

    @staticmethod
    def load(path: Path, input_dim: int) -> "AutoencoderTrainer":
        ckpt = torch.load(str(path), map_location="cpu")
        params = AEParams(**ckpt["params"])  # type: ignore[arg-type]
        trainer = AutoencoderTrainer(input_dim, params)
        trainer.model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
        return trainer
