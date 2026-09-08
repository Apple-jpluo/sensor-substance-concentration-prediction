#!/usr/bin/env python3
"""Evaluate the final SPSM using periods 1--7 as LOPO groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from statsml.config import load_config
from statsml.data import load_datasets
from statsml.evaluation import evaluate_spsm_lopo, fold_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    parser.add_argument(
        "--oof-output",
        type=Path,
        help="Optional row-level OOF output. Keep it local because it includes targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_datasets(args.data_dir)
    config = load_config(args.config)
    fold_metrics, oof, pooled = evaluate_spsm_lopo(
        bundle.labelled, bundle.feature_columns, config
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "spsm_lopo_folds.csv", index=False)
    fold_summary(fold_metrics).to_csv(args.output_dir / "spsm_lopo_fold_summary.csv")
    if args.oof_output is not None:
        args.oof_output.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(args.oof_output, index=False)
    with (args.output_dir / "spsm_lopo_pooled_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pooled, handle, indent=2)


if __name__ == "__main__":
    main()
