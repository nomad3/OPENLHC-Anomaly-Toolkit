from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import uproot  # type: ignore
    import awkward as ak  # type: ignore
except Exception:  # pragma: no cover
    uproot = None  # type: ignore
    ak = None  # type: ignore


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None


@dataclass
class DataConfig:
    dataset_name: str
    root_files: List[str]
    features: List[str]
    split: Dict[str, float]
    cache_parquet: Optional[str] = None
    max_events: Optional[int] = None


def load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_from_parquet(parquet_path: Path, features: List[str], max_events: Optional[int]) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if features:
        df = df[features]
    if max_events is not None:
        df = df.iloc[:max_events]
    return df


def _load_from_root(root_paths: List[str], features: List[str], max_events: Optional[int]) -> pd.DataFrame:
    if uproot is None:
        raise RuntimeError("uproot is not available; cannot read ROOT files.")
    arrays: List[pd.DataFrame] = []
    remaining = max_events
    for p in root_paths:
        with uproot.open(p) as f:  # type: ignore
            # Heuristic: prefer a tree named 'Events' else choose first TTree
            keys = [k for k in f.keys()]
            tree_key = None
            for k in keys:
                if "Events" in k:
                    tree_key = k
                    break
            if tree_key is None:
                # pick the first TTree-like
                for k in keys:
                    try:
                        obj = f[k]
                        if hasattr(obj, "arrays"):
                            tree_key = k
                            break
                    except Exception:
                        continue
            if tree_key is None:
                raise RuntimeError(f"No suitable tree found in ROOT file: {p}")
            tree = f[tree_key]
            arr = tree.arrays(features, library="ak")  # type: ignore
            # Convert Awkward to Pandas (flat branches assumed)
            pdf = ak.to_pandas(arr)  # type: ignore
            if remaining is not None:
                pdf = pdf.iloc[:remaining]
                remaining = max(0, remaining - len(pdf))
            arrays.append(pdf)
            if remaining == 0:
                break
    if not arrays:
        raise RuntimeError("No data loaded from ROOT files")
    df = pd.concat(arrays, ignore_index=True)
    return df


def _generate_synthetic(features: List[str], n: int = 10000, anomaly_frac: float = 0.01, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = len(features)
    # Normal background
    X_b = rng.normal(loc=0.0, scale=1.0, size=(int(n * (1 - anomaly_frac)), d))
    # Anomalies with shifted mean/variance
    X_s = rng.normal(loc=3.0, scale=1.5, size=(int(n * anomaly_frac), d))
    X = np.vstack([X_b, X_s])
    y = np.hstack([np.zeros(len(X_b)), np.ones(len(X_s))]).astype(int)
    rng.shuffle(X)
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=features)
    return df, y


def _train_val_test_split(df: pd.DataFrame, y: Optional[np.ndarray], split: Dict[str, float], seed: int = 42) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(split.get("train", 0.7) * n)
    n_val = int(split.get("val", 0.15) * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    X = df.values.astype(np.float32)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    if y is None:
        return DatasetSplits(X_train, X_val, X_test)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def load_data_from_config(config_path: Path) -> Tuple[DatasetSplits, DataConfig]:
    cfg = load_yaml(config_path)
    data_cfg = DataConfig(
        dataset_name=cfg.get("dataset_name", "dataset"),
        root_files=cfg.get("root_files", []) or [],
        features=cfg.get("features", []),
        split=cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15}),
        cache_parquet=cfg.get("cache_parquet"),
        max_events=cfg.get("max_events"),
    )

    if not data_cfg.features:
        # Default to 5 generic features if not provided
        data_cfg.features = ["f1", "f2", "f3", "f4", "f5"]

    df: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None

    if data_cfg.cache_parquet:
        parquet_path = Path(data_cfg.cache_parquet)
        if parquet_path.exists():
            df = _load_from_parquet(parquet_path, data_cfg.features, data_cfg.max_events)

    if df is None and data_cfg.root_files:
        try:
            df = _load_from_root(data_cfg.root_files, data_cfg.features, data_cfg.max_events)
        except Exception:
            df = None

    if df is None:
        df, y = _generate_synthetic(data_cfg.features, n=10000)

    splits = _train_val_test_split(df, y, data_cfg.split)
    return splits, data_cfg
