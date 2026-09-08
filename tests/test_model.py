import numpy as np
import pandas as pd

from statsml.models import SoftProbabilitySpecialistModel


def toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for substance in (1, 2, 3):
        for period in (1, 2):
            for _ in range(8):
                x1, x2 = rng.normal(loc=substance, scale=0.2, size=2)
                rows.append({
                    "period": period,
                    "substance": substance,
                    "concentration": 10 * substance + x1,
                    "feature_1": x1,
                    "feature_2": x2,
                })
    return pd.DataFrame(rows)


def test_spsm_prediction_shapes_and_probability_alignment():
    frame = toy_frame()
    model = SoftProbabilitySpecialistModel(
        feature_columns=("feature_1", "feature_2"),
        classifier_config={
            "hidden_layer_sizes": [8],
            "solver": "lbfgs",
            "max_iter": 500,
            "alpha": 0.001,
        },
        regressor_config={"n_estimators": 8, "min_samples_leaf": 1, "n_jobs": 1},
        random_state=42,
    ).fit(frame)
    predicted_class, predicted_concentration = model.predict(frame.head(5))
    probabilities, specialist_predictions, classes = model.predict_components(frame.head(5))
    assert predicted_class.shape == (5,)
    assert predicted_concentration.shape == (5,)
    assert probabilities.shape == specialist_predictions.shape == (5, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        predicted_concentration, (probabilities * specialist_predictions).sum(axis=1)
    )
    assert set(classes) == {1, 2, 3}
