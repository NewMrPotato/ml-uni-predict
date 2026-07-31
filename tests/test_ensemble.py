from pathlib import Path

import numpy as np
import pytest

from flexpredict import (
    ConfigurationError,
    EnsembleCompatibilityError,
    EnsemblePredictor,
    PredictionResult,
    Predictor,
)


class ConstantRegressor:
    _estimator_type = "regressor"

    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return np.full(len(values), self.value, dtype=float)


class ProbabilityPredictor:
    def __init__(self, values, classes):
        self.values = np.asarray(values)
        self.classes = np.asarray(classes)

    def predict_proba(self, data):
        return PredictionResult(
            self.values,
            task="classification",
            output_kind="probabilities",
            classes=self.classes,
            is_single=len(self.values) == 1,
        )

    def predict(self, data):
        labels = self.classes[np.argmax(self.values, axis=1)].reshape(-1, 1)
        return PredictionResult(
            labels,
            task="classification",
            output_kind="labels",
            classes=self.classes,
            is_single=len(labels) == 1,
        )


def make_regression_ensemble(**kwargs):
    return EnsemblePredictor(
        [Predictor(ConstantRegressor(10)), Predictor(ConstantRegressor(20))],
        **kwargs,
    )


def test_mean_and_weighted_mean_ensemble():
    mean = make_regression_ensemble()
    weighted = make_regression_ensemble(
        aggregation="weighted_mean",
        aggregation_weights=[0.25, 0.75],
    )

    assert mean.predict([[1, 2]]).single() == 15.0
    assert weighted.predict([[1, 2]]).single() == 17.5


def test_ensemble_loads_global_weights_from_npy(tmp_path: Path):
    path = tmp_path / "global_weights.npy"
    np.save(path, np.array([0.8, 0.2]))

    ensemble = make_regression_ensemble(
        aggregation="weighted_mean",
        aggregation_weights=path,
    )

    assert np.allclose(ensemble.weights, [0.8, 0.2])
    assert ensemble.predict([[0]]).single() == 12.0


@pytest.mark.parametrize(
    "weights",
    [[0, 0], [-1, 2], [np.nan, 1], [1]],
)
def test_ensemble_rejects_invalid_weights(weights):
    with pytest.raises(ConfigurationError):
        make_regression_ensemble(
            aggregation="weighted_mean",
            aggregation_weights=weights,
        )


def test_probability_ensemble_aligns_class_order():
    first = ProbabilityPredictor([[0.8, 0.2]], ["no", "yes"])
    second = ProbabilityPredictor([[0.7, 0.3]], ["yes", "no"])
    ensemble = EnsemblePredictor([first, second])

    result = ensemble.predict_proba(None)

    assert result.classes.tolist() == ["no", "yes"]
    assert np.allclose(result.values, [[0.55, 0.45]])


def test_ensemble_rejects_incompatible_shapes():
    first = ProbabilityPredictor([[0.8, 0.2]], ["no", "yes"])
    second = ProbabilityPredictor([[0.7, 0.2, 0.1]], ["a", "b", "c"])

    with pytest.raises(EnsembleCompatibilityError, match="shape"):
        EnsemblePredictor([first, second]).predict_proba(None)


def test_custom_aggregation_must_preserve_prediction_shape():
    ensemble = make_regression_ensemble(aggregation=lambda values: values[0, 0])

    with pytest.raises(Exception, match="shape"):
        ensemble.predict([[1], [2]])
