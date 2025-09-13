from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import yaml

from .data_loader import load_data_from_config
from .preprocess import fit_transform
from .models.baselines import BaselineIForest, BaselinePCA
from .models.autoencoder import AEParams, AutoencoderTrainer
from .models.vae import VAEParams, VAETrainer
from .utils import RunPaths, ensure_dirs, set_global_seed, setup_logging


def run_training(data_config_path: Path, model_config_path: Path, outdir: Path) -> Dict[str, Any]:
    ensure_dirs()
    setup_logging()

    with open(model_config_path, "r") as f:
        mcfg = yaml.safe_load(f)

    seed = int(mcfg.get("seed", 42))
    set_global_seed(seed)

    splits, data_cfg = load_data_from_config(data_config_path)
    X_train_s, X_val_s, X_test_s, prep = fit_transform(splits.X_train, splits.X_val, splits.X_test)

    model_name = str(mcfg.get("model", "autoencoder")).lower()

    artifacts_dir = RunPaths.create(outdir)
    # Save configs for reproducibility
    (artifacts_dir.run_dir / "data_config.yaml").write_text(Path(data_config_path).read_text())
    (artifacts_dir.run_dir / "model_config.yaml").write_text(Path(model_config_path).read_text())

    joblib.dump(prep.scaler, artifacts_dir.run_dir / "scaler.joblib")

    model_info: Dict[str, Any] = {"model": model_name}

    if model_name in {"pca"}:
        params = mcfg.get("pca", {})
        model = BaselinePCA(n_components=int(params.get("n_components", 4)))
        model.fit(X_train_s)
        joblib.dump(model, artifacts_dir.run_dir / "model.joblib")
    elif model_name in {"iforest", "isolationforest"}:
        params = mcfg.get("iforest", {})
        model = BaselineIForest(
            n_estimators=int(params.get("n_estimators", 200)),
            max_samples=params.get("max_samples", "auto"),
            contamination=float(params.get("contamination", 0.01)),
        )
        model.fit(X_train_s)
        joblib.dump(model, artifacts_dir.run_dir / "model.joblib")
    elif model_name in {"ae", "autoencoder"}:
        params = mcfg.get("ae", {})
        ae_params = AEParams(
            hidden_dims=list(map(int, params.get("hidden_dims", [64, 32, 16, 32, 64]))),
            activation=str(params.get("activation", "relu")),
            dropout=float(params.get("dropout", 0.0)),
            lr=float(params.get("lr", 1e-3)),
            batch_size=int(params.get("batch_size", 256)),
            epochs=int(params.get("epochs", 20)),
            device=str(params.get("device", "cpu")),
        )
        trainer = AutoencoderTrainer(input_dim=X_train_s.shape[1], params=ae_params)
        trainer.fit(X_train_s, X_val_s)
        from pathlib import Path as _P
        trainer.save(_P(artifacts_dir.run_dir / "model.pt"))
    elif model_name in {"vae"}:
        params = mcfg.get("vae", {})
        vae_params = VAEParams(
            hidden_dims=list(map(int, params.get("hidden_dims", [64, 32]))),
            latent_dim=int(params.get("latent_dim", 8)),
            beta=float(params.get("beta", 1.0)),
            lr=float(params.get("lr", 1e-3)),
            batch_size=int(params.get("batch_size", 256)),
            epochs=int(params.get("epochs", 30)),
            device=str(params.get("device", "cpu")),
        )
        trainer = VAETrainer(input_dim=X_train_s.shape[1], params=vae_params)
        trainer.fit(X_train_s)
        import torch
        torch.save({
            "state_dict": trainer.model.state_dict(),
            "params": vae_params.__dict__,
        }, str(artifacts_dir.run_dir / "model.pt"))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Save minimal training metadata
    (artifacts_dir.run_dir / "meta.json").write_text(json.dumps({
        "dataset": data_cfg.dataset_name,
        "features": data_cfg.features,
        "seed": seed,
        "model": model_name,
    }, indent=2))

    return {
        "run_dir": str(artifacts_dir.run_dir),
        "model": model_name,
        "features": data_cfg.features,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    run_training(args.data_config, args.model_config, args.outdir)


if __name__ == "__main__":
    main()
