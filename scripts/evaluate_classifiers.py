#!/usr/bin/env python3
"""Re-evaluate the principal substance-classification baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsml.config import load_config
from statsml.data import load_datasets
from statsml.metrics import classification_metrics
from statsml.models import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    parser.add_argument("--models", default="logistic,knn,random_forest,mlp")
    return parser.parse_args()


def build_models(config: dict) -> dict:
    seed = int(config["random_state"])
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", p=2)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "mlp": build_classifier(config["classifier"], seed),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_datasets(args.data_dir)
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    available = build_models(config)
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choose from {sorted(available)}")

    feature_columns = list(bundle.feature_columns)
    x = bundle.train.loc[:, feature_columns]
    y = bundle.train["substance"].astype(int)
    groups = bundle.train["period"]
    fold_rows = []
    summaries = {}

    for name in selected:
        model = available[name]
        oof = pd.Series(index=bundle.train.index, dtype=int)
        for train_index, validation_index in LeaveOneGroupOut().split(x, y, groups):
            fitted = clone(model).fit(x.iloc[train_index], y.iloc[train_index])
            prediction = fitted.predict(x.iloc[validation_index]).astype(int)
            oof.iloc[validation_index] = prediction
            fold_rows.append({
                "model": name,
                "held_out_period": int(groups.iloc[validation_index].iloc[0]),
                **classification_metrics(y.iloc[validation_index], prediction),
            })

        final_model = clone(model).fit(x, y)
        public_prediction = final_model.predict(bundle.public.loc[:, feature_columns]).astype(int)
        model_folds = pd.DataFrame(fold_rows).query("model == @name")
        summaries[name] = {
            "lopo_unweighted_fold_mean": model_folds[["accuracy", "macro_f1", "balanced_accuracy"]].mean().to_dict(),
            "lopo_pooled_oof": classification_metrics(y, oof.astype(int)),
            "public_holdout": classification_metrics(bundle.public["substance"], public_prediction),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "classification_lopo_folds.csv", index=False)
    with (args.output_dir / "classification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
