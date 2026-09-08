"""Temporal validation procedures used by the paper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from .metrics import joint_metrics, summarize_folds
from .models import SoftProbabilitySpecialistModel


def evaluate_spsm_lopo(
    labelled: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run LOPO and return fold metrics, row-level OOF predictions and pooled metrics.

    Both aggregation definitions are returned so that period weighting remains
    explicit.
    """
    splitter = LeaveOneGroupOut()
    groups = labelled["period"]
    fold_rows: list[dict[str, float | int]] = []
    prediction_parts: list[pd.DataFrame] = []

    for train_index, validation_index in splitter.split(labelled, groups=groups):
        train_fold = labelled.iloc[train_index]
        validation_fold = labelled.iloc[validation_index]
        held_out_period = int(validation_fold["period"].iloc[0])

        model = SoftProbabilitySpecialistModel(
            feature_columns=feature_columns,
            classifier_config=config["classifier"],
            regressor_config=config["regressor"],
            random_state=int(config["random_state"]),
        ).fit(train_fold)
        predicted_substance, predicted_concentration = model.predict(validation_fold)

        metrics = joint_metrics(
            validation_fold["substance"],
            predicted_substance,
            validation_fold["concentration"],
            predicted_concentration,
        )
        fold_rows.append({"held_out_period": held_out_period, **metrics})
        prediction_parts.append(
            pd.DataFrame(
                {
                    "row_index": validation_index,
                    "period": validation_fold["period"].to_numpy(),
                    "true_substance": validation_fold["substance"].to_numpy(),
                    "predicted_substance": predicted_substance,
                    "true_concentration": validation_fold["concentration"].to_numpy(),
                    "predicted_concentration": predicted_concentration,
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows).sort_values("held_out_period")
    oof = pd.concat(prediction_parts, ignore_index=True).sort_values("row_index")
    if not np.array_equal(oof["row_index"].to_numpy(), np.arange(len(labelled))):
        raise RuntimeError("LOPO predictions do not cover every labelled row exactly once")

    pooled = joint_metrics(
        oof["true_substance"],
        oof["predicted_substance"],
        oof["true_concentration"],
        oof["predicted_concentration"],
    )
    return fold_metrics, oof, pooled


def fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return summarize_folds(fold_metrics)
