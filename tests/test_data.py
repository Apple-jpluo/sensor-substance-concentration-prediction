import pandas as pd
import pytest

from statsml.data import discover_feature_columns, validate_frame


def test_feature_columns_are_numeric_ordered():
    frame = pd.DataFrame({"feature_10": [0.0], "period": [1], "feature_2": [1.0]})
    assert discover_feature_columns(frame) == ("feature_2", "feature_10")


def test_private_target_leak_is_detectable():
    frame = pd.DataFrame({
        "period": [8],
        "substance": [1],
        "concentration": [10.0],
        "feature_1": [0.0],
    })
    with pytest.raises(ValueError, match="private"):
        validate_frame(
            frame,
            name="private_test.csv",
            feature_columns=("feature_1",),
            labelled=False,
        )


def test_validation_rejects_missing_values():
    frame = pd.DataFrame({
        "period": [1], "substance": [1], "concentration": [5.0], "feature_1": [None]
    })
    with pytest.raises(ValueError, match="missing values"):
        validate_frame(frame, name="toy", feature_columns=("feature_1",), labelled=True)
