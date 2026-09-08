#!/usr/bin/env python3
"""Generate the compact EDA needed to motivate temporal validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsml.data import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    return parser.parse_args()


def save_distribution_figures(labelled: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labelled["substance"].astype(int).value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set(title="Rows by substance", xlabel="Substance", ylabel="Count")
    labelled["period"].value_counts().sort_index().plot.bar(ax=axes[1])
    axes[1].set(title="Rows by period", xlabel="Period", ylabel="Count")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_counts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labelled.boxplot(column="concentration", by="substance", ax=axes[0])
    axes[0].set(title="Concentration by substance", xlabel="Substance", ylabel="Concentration")
    labelled.boxplot(column="concentration", by="period", ax=axes[1])
    axes[1].set(title="Concentration by period", xlabel="Period", ylabel="Concentration")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "concentration_shift.png", dpi=180)
    plt.close(fig)


def save_pca(labelled: pd.DataFrame, features: tuple[str, ...], output_dir: Path) -> None:
    scores = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(labelled.loc[:, list(features)])
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for value in sorted(labelled["substance"].astype(int).unique()):
        mask = labelled["substance"].to_numpy(dtype=int) == value
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=4, alpha=0.35, label=str(value))
    for value in sorted(labelled["period"].unique()):
        mask = labelled["period"].to_numpy() == value
        axes[1].scatter(scores[mask, 0], scores[mask, 1], s=4, alpha=0.35, label=str(value))
    axes[0].set(title="PCA coloured by substance", xlabel="PC1", ylabel="PC2")
    axes[1].set(title="PCA coloured by period", xlabel="PC1", ylabel="PC2")
    axes[0].legend(title="Substance", markerscale=3, ncol=2)
    axes[1].legend(title="Period", markerscale=3, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_substance_and_period.png", dpi=180)
    plt.close(fig)


def save_drift_ranking(labelled: pd.DataFrame, features: tuple[str, ...], output_dir: Path) -> None:
    period_means = labelled.groupby("period")[list(features)].mean()
    scale = labelled.loc[:, list(features)].std().replace(0, np.nan)
    normalized_range = (period_means.max() - period_means.min()).div(scale).fillna(0)
    ranking = normalized_range.sort_values(ascending=False).rename("normalized_mean_range")
    ranking.to_csv(output_dir.parent / "tables" / "feature_drift_ranking.csv")
    top = ranking.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot.barh(ax=ax)
    ax.set(title="Features with strongest between-period mean shift", xlabel="Mean range / global SD")
    fig.tight_layout()
    fig.savefig(output_dir / "top_feature_drift.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bundle = load_datasets(args.data_dir)
    labelled = bundle.labelled
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir.parent / "tables").mkdir(parents=True, exist_ok=True)

    save_distribution_figures(labelled, args.output_dir)
    save_pca(labelled, bundle.feature_columns, args.output_dir)
    save_drift_ranking(labelled, bundle.feature_columns, args.output_dir)

    summary = {
        "train_shape": list(bundle.train.shape),
        "public_shape": list(bundle.public.shape),
        "private_shape": list(bundle.private.shape),
        "n_features": len(bundle.feature_columns),
        "train_period_counts": bundle.train["period"].value_counts().sort_index().to_dict(),
        "train_substance_counts": bundle.train["substance"].value_counts().sort_index().to_dict(),
    }
    with (args.output_dir.parent / "tables" / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
