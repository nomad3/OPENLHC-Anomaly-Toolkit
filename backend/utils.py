from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - allow non-torch environments
    torch = None  # type: ignore


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_RUNS_DIR = DEFAULT_RESULTS_DIR / "runs"
DEFAULT_PLOTS_DIR = DEFAULT_RESULTS_DIR / "plots"
DEFAULT_LEADERBOARD = DEFAULT_RESULTS_DIR / "leaderboard.json"


def ensure_dirs() -> None:
    for d in [DEFAULT_RESULTS_DIR, DEFAULT_RUNS_DIR, DEFAULT_PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints: Path
    logs: Path

    @staticmethod
    def create(root: Path) -> "RunPaths":
        run_dir = root
        checkpoints = run_dir / "checkpoints"
        logs = run_dir / "logs"
        for d in [run_dir, checkpoints, logs]:
            d.mkdir(parents=True, exist_ok=True)
        return RunPaths(run_dir, checkpoints, logs)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def append_leaderboard(entry: Dict[str, Any], leaderboard_path: Path = DEFAULT_LEADERBOARD) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard_path.exists():
        try:
            with leaderboard_path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    with leaderboard_path.open("w") as f:
        json.dump(data, f, indent=2)


def getenv_path(var: str, default: Optional[str] = None) -> Optional[Path]:
    val = os.getenv(var, default)
    return Path(val) if val else None
