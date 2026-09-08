"""Final Soft Probability Specialist Model (SPSM)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(config: Mapping[str, Any], random_state: int) -> Pipeline:
    parameters = dict(config)
    if "hidden_layer_sizes" in parameters:
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(random_state=random_state, **parameters),
            ),
        ]
    )


def build_regressor(
    config: Mapping[str, Any], random_state: int
) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=random_state, **dict(config))


class SoftProbabilitySpecialistModel:
    """MLP classifier plus one concentration regressor per substance.

    Concentration is the probability-weighted sum of specialist predictions:
    ``sum_k P(substance=k|x) * regressor_k(x)``.
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str],
        classifier_config: Mapping[str, Any],
        regressor_config: Mapping[str, Any],
        random_state: int = 42,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.classifier_config = dict(classifier_config)
        self.regressor_config = dict(regressor_config)
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame) -> "SoftProbabilitySpecialistModel":
        x = frame.loc[:, list(self.feature_columns)]
        y_substance = frame["substance"].astype(int)
        self.classifier_ = build_classifier(
            self.classifier_config, self.random_state
        )
        self.classifier_.fit(x, y_substance)

        classes = self.classifier_.named_steps["classifier"].classes_.astype(int)
        self.classes_ = classes
        self.regressors_: dict[int, RandomForestRegressor] = {}
        for substance in classes:
            rows = frame[frame["substance"].astype(int) == substance]
            if rows.empty:
                raise ValueError(f"No training rows for substance {substance}")
            regressor = build_regressor(self.regressor_config, self.random_state)
            regressor.fit(
                rows.loc[:, list(self.feature_columns)], rows["concentration"]
            )
            self.regressors_[int(substance)] = regressor
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "classifier_") or not hasattr(self, "regressors_"):
            raise RuntimeError("The model must be fitted before prediction")

    def predict_components(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        x = frame.loc[:, list(self.feature_columns)]
        probabilities = self.classifier_.predict_proba(x)
        specialist_predictions = np.column_stack(
            [self.regressors_[int(label)].predict(x) for label in self.classes_]
        )
        return probabilities, specialist_predictions, self.classes_.copy()

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        probabilities, specialist_predictions, classes = self.predict_components(frame)
        predicted_substance = classes[np.argmax(probabilities, axis=1)]
        predicted_concentration = np.sum(
            probabilities * specialist_predictions, axis=1
        )
        return predicted_substance.astype(int), predicted_concentration
