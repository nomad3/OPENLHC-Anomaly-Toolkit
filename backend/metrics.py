from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn import metrics as skm


@dataclass
class MetricResults:
    auc: Optional[float]
    average_precision: Optional[float]


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        if y_true is None or y_score is None:
            return None
        if len(np.unique(y_true)) < 2:
            return None
        return float(skm.roc_auc_score(y_true, y_score))
    except Exception:
        return None


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        if y_true is None or y_score is None:
            return None
        if len(np.unique(y_true)) < 2:
            return None
        return float(skm.average_precision_score(y_true, y_score))
    except Exception:
        return None


def compute_basic_metrics(y_true: Optional[np.ndarray], y_score: np.ndarray) -> MetricResults:
    if y_true is None:
        return MetricResults(auc=None, average_precision=None)
    return MetricResults(
        auc=compute_roc_auc(y_true, y_score),
        average_precision=compute_average_precision(y_true, y_score),
    )


def discovery_significance(signal: float, background: float, sigma_b: Optional[float] = None) -> float:
    """
    Compute discovery significance Z_bi for counting experiments.

    If sigma_b is None, use the simple approximation Z = s / sqrt(b) (with small safeguards).
    If sigma_b is provided (absolute background uncertainty), use the profile likelihood
    approximation (Cowan et al., 2011) for discovery significance.
    """
    s = max(0.0, float(signal))
    b = max(0.0, float(background))
    if b <= 0.0:
        return 0.0

    if sigma_b is None or sigma_b <= 0.0:
        return float(s / np.sqrt(b + 1e-12))

    # Cowan profile likelihood approximation
    sigma2 = float(sigma_b) ** 2
    term1 = (s + b) * np.log((s + b) * (b + sigma2) / (b * b + (s + b) * sigma2 + 1e-12) + 1e-12)
    term2 = (b * b / (sigma2 + 1e-12)) * np.log(1.0 + (sigma2 * s) / (b * (b + sigma2) + 1e-12) + 1e-12)
    z2 = 2.0 * (term1 - term2)
    z2 = max(0.0, float(z2))
    return float(np.sqrt(z2))
