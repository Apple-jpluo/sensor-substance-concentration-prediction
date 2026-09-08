#!/usr/bin/env python3
"""Fit SPSM on periods 1--7 and create a period-8 submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from statsml.config import load_config
from statsml.data import load_datasets
from statsml.models import SoftProbabilitySpecialistModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/private_test_predictions_SPSM.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_datasets(args.data_dir)
    config = load_config(args.config)
    model = SoftProbabilitySpecialistModel(
        feature_columns=bundle.feature_columns,
        classifier_config=config["classifier"],
        regressor_config=config["regressor"],
        random_state=int(config["random_state"]),
    ).fit(bundle.labelled)
    predicted_substance, predicted_concentration = model.predict(bundle.private)
    submission = pd.DataFrame({
        "id": np.arange(len(bundle.private)),
        "substance": predicted_substance.astype(int),
        "concentration": predicted_concentration,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()

