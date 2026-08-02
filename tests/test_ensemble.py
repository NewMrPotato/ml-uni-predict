from pathlib import Path

import numpy as np
import pytest

from flexpredict import (
    ConfigurationError,
    EnsembleCompatibilityError,
    EnsembleInferenceError,
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


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("median", 15.0), ("min", 10.0), ("max", 20.0)],
)
def test_other_numeric_aggregations(aggregation, expected):
    ensemble = make_regression_ensemble(aggregation=aggregation)

    assert ensemble.predict([[0]]).single() == expected


def test_ensemble_loads_global_weights_from_npy(tmp_path: Path):
    path = tmp_path / "global_weights.npy"
    np.save(path, np.array([0.8, 0.2]))

    ensemble = make_regression_ensemble(
        aggregation="weighted_mean",
        aggregation_weights=path,
    )

    assert np.allclose(ensemble.weights, [0.8, 0.2])
    assert ensemble.aggregation_weights is ensemble.weights
    assert ensemble.weights.flags.writeable is False
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


def test_ensemble_rejects_invalid_weight_files(tmp_path):
    text_path = tmp_path / "weights.txt"
    text_path.write_text("1, 2")
    missing = tmp_path / "missing.npy"
    corrupt = tmp_path / "corrupt.npy"
    corrupt.write_bytes(b"not-numpy")

    with pytest.raises(ConfigurationError, match=r"\.npy"):
        make_regression_ensemble(
            aggregation="weighted_mean", aggregation_weights=text_path
        )
    with pytest.raises(ConfigurationError, match="does not exist"):
        make_regression_ensemble(
            aggregation="weighted_mean", aggregation_weights=missing
        )
    with pytest.raises(ConfigurationError, match="Could not load"):
        make_regression_ensemble(
            aggregation="weighted_mean", aggregation_weights=corrupt
        )


def test_probability_ensemble_aligns_class_order():
    first = ProbabilityPredictor([[0.8, 0.2]], ["no", "yes"])
    second = ProbabilityPredictor([[0.7, 0.3]], ["yes", "no"])
    ensemble = EnsemblePredictor([first, second])

    result = ensemble.predict_proba(None)

    assert result.classes.tolist() == ["no", "yes"]
    assert np.allclose(result.values, [[0.55, 0.45]])


def test_voting_ensemble_aggregates_labels():
    first = ProbabilityPredictor([[0.8, 0.2], [0.1, 0.9]], ["no", "yes"])
    second = ProbabilityPredictor([[0.7, 0.3], [0.2, 0.8]], ["no", "yes"])
    third = ProbabilityPredictor([[0.1, 0.9], [0.8, 0.2]], ["no", "yes"])

    result = EnsemblePredictor([first, second, third], aggregation="voting").predict(None)

    assert result.values[:, 0].tolist() == ["no", "yes"]


def test_voting_ties_are_resolved_by_predictor_order():
    first = ProbabilityPredictor([[0.8, 0.2]], ["no", "yes"])
    second = ProbabilityPredictor([[0.2, 0.8]], ["no", "yes"])

    result = EnsemblePredictor([first, second], aggregation="voting").predict(None)

    assert result.single() == "no"


def test_ensemble_rejects_incompatible_shapes():
    first = ProbabilityPredictor([[0.8, 0.2]], ["no", "yes"])
    second = ProbabilityPredictor([[0.7, 0.2, 0.1]], ["a", "b", "c"])

    with pytest.raises(EnsembleCompatibilityError, match="shape"):
        EnsemblePredictor([first, second]).predict_proba(None)


def test_ensemble_rejects_incompatible_metadata():
    class StaticPredictor:
        def __init__(self, result):
            self.result = result

        def predict(self, data):
            return self.result

    regression = PredictionResult(np.array([[1.0]]), task="regression")
    classification = PredictionResult(
        np.array([["yes"]]), task="classification", output_kind="labels"
    )
    probabilities = PredictionResult(
        np.array([[0.2, 0.8]]), task="classification", output_kind="probabilities"
    )

    with pytest.raises(EnsembleCompatibilityError, match="task"):
        EnsemblePredictor(
            [StaticPredictor(regression), StaticPredictor(classification)]
        ).predict(None)
    with pytest.raises(EnsembleCompatibilityError, match="returned"):
        EnsemblePredictor(
            [StaticPredictor(classification), StaticPredictor(probabilities)]
        ).predict(None)


def test_ensemble_constructor_validates_members_and_aggregation():
    with pytest.raises(ConfigurationError, match="at least one"):
        EnsemblePredictor([])
    with pytest.raises(ConfigurationError, match="must define predict"):
        EnsemblePredictor([object()])
    with pytest.raises(ConfigurationError, match="Unknown aggregation"):
        make_regression_ensemble(aggregation="unknown")
    with pytest.raises(ConfigurationError, match="only with"):
        make_regression_ensemble(aggregation="mean", aggregation_weights=[1, 1])
    with pytest.raises(ConfigurationError, match="custom aggregation"):
        make_regression_ensemble(
            aggregation=lambda values: values.mean(axis=0),
            aggregation_weights=[1, 1],
        )


def test_ensemble_identifies_a_failing_member():
    class FailingPredictor:
        name = "broken-model"

        def predict(self, data):
            raise RuntimeError("device unavailable")

    ensemble = EnsemblePredictor(
        [Predictor(ConstantRegressor(10), name="healthy"), FailingPredictor()]
    )

    with pytest.raises(EnsembleInferenceError, match="broken-model.*device unavailable"):
        ensemble.predict([[0]])


def test_ensemble_rejects_inconsistent_single_input_metadata():
    class StaticPredictor:
        def __init__(self, is_single):
            self.is_single = is_single

        def predict(self, data):
            return PredictionResult(
                np.array([[1.0]]), task="regression", is_single=self.is_single
            )

    ensemble = EnsemblePredictor([StaticPredictor(True), StaticPredictor(False)])

    with pytest.raises(EnsembleCompatibilityError, match="is_single"):
        ensemble.predict(None)


def test_custom_aggregation_must_preserve_prediction_shape():
    ensemble = make_regression_ensemble(aggregation=lambda values: values[0, 0])

    with pytest.raises(Exception, match="shape"):
        ensemble.predict([[1], [2]])
