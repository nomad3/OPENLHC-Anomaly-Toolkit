from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import metrics as skm
import matplotlib.pyplot as plt

from .data_loader import load_data_from_config
from .metrics import MetricResults, compute_basic_metrics
from .utils import DEFAULT_LEADERBOARD, RunPaths, append_leaderboard, ensure_dirs, setup_logging


def _load_model(run_dir: Path, input_dim: int):
    # Try torch first
    pt_path = run_dir / "model.pt"
    if pt_path.exists():
        try:
            from .models.autoencoder import AutoencoderTrainer, AEParams
            ckpt = None
            return ("torch", pt_path)
        except Exception:
            pass
    # Fallback to joblib
    jl = run_dir / "model.joblib"
    if jl.exists():
        model = joblib.load(jl)
        return ("sklearn", model)
    raise FileNotFoundError("No model artifact found in run directory")


def _score(model_ref, X: np.ndarray, input_dim: int) -> np.ndarray:
    kind, obj = model_ref
    if kind == "torch":
        # Inspect model_config to distinguish VAE/AE? For simplicity, try AE first
        from .models.autoencoder import AutoencoderTrainer
        trainer = AutoencoderTrainer.load(obj, input_dim)  # type: ignore[arg-type]
        return trainer.score(X)
    else:
        model = obj
        return model.score(X)


def run_evaluation(run_dir: Path, outdir: Path, leaderboard_path: Optional[Path] = None) -> Dict[str, Any]:
    ensure_dirs()
    setup_logging()

    run_dir = Path(run_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_cfg_path = run_dir / "data_config.yaml"
    if not data_cfg_path.exists():
        raise FileNotFoundError("data_config.yaml not found in run directory")

    splits, data_cfg = load_data_from_config(data_cfg_path)

    scaler_path = run_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path)
    X_train_s = scaler.transform(splits.X_train)
    X_val_s = scaler.transform(splits.X_val)
    X_test_s = scaler.transform(splits.X_test)

    model_ref = _load_model(run_dir, input_dim=X_train_s.shape[1])
    scores = _score(model_ref, X_test_s, input_dim=X_train_s.shape[1])

    y_true = splits.y_test if splits.y_test is not None else None
    m: MetricResults = compute_basic_metrics(y_true, scores)

    # Save CSV of scores
    df = pd.DataFrame({"score": scores})
    if y_true is not None:
        df["label"] = y_true
    df.to_csv(outdir / "scores.csv", index=False)

    # Save simple plots
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=50, alpha=0.8)
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "score_hist.png", dpi=150)
    plt.close()

    if y_true is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = skm.roc_curve(y_true, scores)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUC={m.auc:.3f}")
        plt.plot([0,1],[0,1],"k--", alpha=0.5)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "roc.png", dpi=150)
        plt.close()

    # Leaderboard entry
    entry = {
        "dataset": data_cfg.dataset_name,
        "model": json.loads((run_dir / "meta.json").read_text()).get("model", "model"),
        "metrics": {"auc": m.auc, "ap": m.average_precision},
    }
    append_leaderboard(entry, leaderboard_path or DEFAULT_LEADERBOARD)

    return {
        "outdir": str(outdir),
        "metrics": entry["metrics"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--append-leaderboard", type=Path, default=None)
    args = p.parse_args()

    run_evaluation(args.run_dir, args.outdir, args.append_leaderboard)


if __name__ == "__main__":
    main()
