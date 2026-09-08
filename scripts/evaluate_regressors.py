#!/usr/bin/env python3
"""Compare the paper's key global and oracle-conditional regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsml.config import load_config
from statsml.data import load_datasets
from statsml.metrics import regression_metrics
from statsml.models import build_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    return parser.parse_args()


def predict_conditional_rf(train: pd.DataFrame, validation: pd.DataFrame, features: tuple[str, ...], config: dict) -> np.ndarray:
    prediction = np.empty(len(validation), dtype=float)
    for substance in sorted(validation["substance"].unique()):
        train_rows = train[train["substance"] == substance]
        validation_mask = validation["substance"].to_numpy() == substance
        if train_rows.empty:
            raise ValueError(f"Training fold has no rows for substance {substance}")
        model = build_regressor(config["regressor"], int(config["random_state"]))
        model.fit(train_rows.loc[:, list(features)], train_rows["concentration"])
        prediction[validation_mask] = model.predict(
            validation.loc[validation_mask, list(features)]
        )
    return prediction


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_datasets(args.data_dir)
    train = bundle.train
    features = bundle.feature_columns
    global_knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance", p=2))
    rows = []
    oof = {"global_knn": np.empty(len(train)), "oracle_conditional_rf": np.empty(len(train))}

    splitter = LeaveOneGroupOut()
    for train_index, validation_index in splitter.split(train, groups=train["period"]):
        train_fold = train.iloc[train_index]
        validation_fold = train.iloc[validation_index]
        period = int(validation_fold["period"].iloc[0])

        model = clone(global_knn).fit(
            train_fold.loc[:, list(features)], train_fold["concentration"]
        )
        global_prediction = model.predict(validation_fold.loc[:, list(features)])
        conditional_prediction = predict_conditional_rf(train_fold, validation_fold, features, config)
        oof["global_knn"][validation_index] = global_prediction
        oof["oracle_conditional_rf"][validation_index] = conditional_prediction
        rows.extend([
            {"model": "global_knn", "held_out_period": period, **regression_metrics(validation_fold["concentration"], global_prediction)},
            {"model": "oracle_conditional_rf", "held_out_period": period, **regression_metrics(validation_fold["concentration"], conditional_prediction)},
        ])

    public_global_model = clone(global_knn).fit(
        train.loc[:, list(features)], train["concentration"]
    )
    public_predictions = {
        "global_knn": public_global_model.predict(
            bundle.public.loc[:, list(features)]
        ),
        "oracle_conditional_rf": predict_conditional_rf(train, bundle.public, features, config),
    }
    fold_frame = pd.DataFrame(rows)
    summary = {}
    for name in oof:
        model_folds = fold_frame.query("model == @name")
        summary[name] = {
            "lopo_unweighted_fold_mean": model_folds[["mae", "rmse", "r2"]].mean().to_dict(),
            "lopo_pooled_oof": regression_metrics(train["concentration"], oof[name]),
            "public_holdout": regression_metrics(bundle.public["concentration"], public_predictions[name]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(args.output_dir / "regression_lopo_folds.csv", index=False)
    with (args.output_dir / "regression_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
