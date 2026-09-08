"""Dataset loading and schema validation.

The model must never use ``period`` as a feature. It is retained only as the
group label for Leave-One-Period-Out validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ("substance", "concentration")
PERIOD_COLUMN = "period"


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    public: pd.DataFrame
    private: pd.DataFrame
    feature_columns: tuple[str, ...]

    @property
    def labelled(self) -> pd.DataFrame:
        """Return periods 1--7 with a fresh, unique row index."""
        return pd.concat([self.train, self.public], ignore_index=True)


def _feature_number(name: str) -> int:
    try:
        return int(name.removeprefix("feature_"))
    except ValueError as exc:
        raise ValueError(f"Malformed feature name: {name!r}") from exc


def discover_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(
        sorted(
            (column for column in frame.columns if column.startswith("feature_")),
            key=_feature_number,
        )
    )
    if not columns:
        raise ValueError("No columns named feature_<number> were found")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature names were found")
    return columns


def validate_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    feature_columns: tuple[str, ...],
    labelled: bool,
) -> None:
    if not labelled:
        leaked_targets = set(TARGET_COLUMNS).intersection(frame.columns)
        if leaked_targets:
            raise ValueError(
                f"{name} unexpectedly contains private targets: {sorted(leaked_targets)}"
            )
    required = {PERIOD_COLUMN, *feature_columns}
    if labelled:
        required.update(TARGET_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    checked = [PERIOD_COLUMN, *feature_columns]
    if labelled:
        checked.extend(TARGET_COLUMNS)
    if frame[checked].isna().any().any():
        raise ValueError(f"{name} contains missing values in modelling columns")

    numeric = frame[list(feature_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains non-finite feature values")

    if labelled:
        substances = frame["substance"].to_numpy(dtype=float)
        if not np.equal(substances, substances.astype(int)).all():
            raise ValueError(f"{name}.substance must contain integer class labels")


def load_datasets(data_dir: str | Path) -> DatasetBundle:
    """Load the three original CSV files from ``data_dir``.

    Expected files are ``train.csv``, ``public_test.csv`` and
    ``private_test.csv``. The private file must not contain target columns.
    """
    data_dir = Path(data_dir)
    paths = {
        "train": data_dir / "train.csv",
        "public": data_dir / "public_test.csv",
        "private": data_dir / "private_test.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing dataset files:\n- " + "\n- ".join(missing_files)
        )

    train = pd.read_csv(paths["train"])
    public = pd.read_csv(paths["public"])
    private = pd.read_csv(paths["private"])
    feature_columns = discover_feature_columns(train)

    validate_frame(
        train, name="train.csv", feature_columns=feature_columns, labelled=True
    )
    validate_frame(
        public,
        name="public_test.csv",
        feature_columns=feature_columns,
        labelled=True,
    )
    validate_frame(
        private,
        name="private_test.csv",
        feature_columns=feature_columns,
        labelled=False,
    )

    for name, frame in (("public", public), ("private", private)):
        other_features = discover_feature_columns(frame)
        if other_features != feature_columns:
            raise ValueError(f"{name} feature schema does not match train.csv")
    return DatasetBundle(train, public, private, feature_columns)
