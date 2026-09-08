"""Metric definitions shared by all experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def regression_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def joint_metrics(
    true_substance: Iterable,
    predicted_substance: Iterable,
    true_concentration: Iterable,
    predicted_concentration: Iterable,
) -> dict[str, float]:
    return {
        **classification_metrics(true_substance, predicted_substance),
        **regression_metrics(true_concentration, predicted_concentration),
    }


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return unweighted mean and sample SD across held-out periods."""
    metric_columns = [column for column in fold_metrics if column != "held_out_period"]
    return fold_metrics[metric_columns].agg(["mean", "std"]).T

