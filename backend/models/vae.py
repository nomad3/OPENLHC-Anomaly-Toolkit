from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for the VAE model.")


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int = 8, activation: str = "relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        Act = acts.get(activation, nn.ReLU)
        dims = [input_dim] + hidden_dims
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i+1]), Act()]
        self.encoder = nn.Sequential(*enc_layers)
        h = hidden_dims[-1] if hidden_dims else input_dim
        self.mu = nn.Linear(h, latent_dim)
        self.logvar = nn.Linear(h, latent_dim)
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers += [nn.Linear(dec_dims[i], dec_dims[i+1])]
            if i < len(dec_dims) - 2:
                dec_layers += [Act()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decoder(z)
        return xr, mu, logvar


def vae_loss(x: torch.Tensor, xr: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    rec = ((x - xr) ** 2).sum(dim=1).mean()
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld, rec.detach(), kld.detach()


@dataclass
class VAEParams:
    hidden_dims: List[int]
    latent_dim: int = 8
    beta: float = 1.0
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    device: str = "cpu"


class VAETrainer:
    def __init__(self, input_dim: int, params: VAEParams):
        self.params = params
        self.device = torch.device(params.device)
        self.model = VAE(input_dim, params.hidden_dims, params.latent_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=params.lr)

    def fit(self, X_train: np.ndarray) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.params.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.params.epochs):
            for (xb,) in dl:
                xb = xb.to(self.device)
                self.opt.zero_grad()
                xr, mu, logvar = self.model(xb)
                loss, _, _ = vae_loss(xb, xr, mu, logvar, beta=self.params.beta)
                loss.backward()
                self.opt.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            xr, mu, logvar = self.model(x)
            rec = ((x - xr) ** 2).sum(dim=1)
            s = rec + self.params.beta * (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).sum(dim=1)
            return s.cpu().numpy().astype(np.float32)
