import numpy as np
import pytest

from flexpredict import (
    ConfigurationError,
    EnsemblePredictor,
    InputSchema,
    MissingFeatureError,
    Predictor,
    PreprocessingError,
    Standardizer,
    UnexpectedFeatureError,
)


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


class ListRegressor:
    def predict(self, values):
        return [float(row[0]) for row in values]


class DifferenceRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        array = np.asarray(values, dtype=float)
        return array[:, 0] - array[:, 1]


class BinaryClassifier:
    _estimator_type = "classifier"
    classes_ = np.array(["no", "yes"])

    def predict(self, values):
        return np.where(np.asarray(values)[:, 0] > 0, "yes", "no")

    def predict_proba(self, values):
        probability = np.clip(np.asarray(values, dtype=float)[:, 0], 0, 1)
        return np.column_stack([1 - probability, probability])


class ForwardProbabilityClassifier:
    def predict(self, values):
        probability = np.asarray(values, dtype=float)[:, 0]
        return np.column_stack([1 - probability, probability])


def test_predictor_zero_config_accepts_arrays():
    predictor = Predictor(SumRegressor())

    single = predictor.predict([1, 2, 3])
    batch = predictor.predict([[1, 2], [3, 4]])

    assert single.values.shape == (1, 1)
    assert single.single() == 6.0
    assert np.allclose(batch.values, [[3], [7]])
    assert batch.task == "regression"


def test_predictor_features_enable_all_named_formats():
    predictor = Predictor(SumRegressor(), features=["x1", "x2"])

    single = predictor.predict({"x2": 2, "x1": 1})
    columns = predictor.predict({"x1": [1, 3], "x2": [2, 4]})
    records = predictor.predict([{"x1": 1, "x2": 2}, {"x1": 3, "x2": 4}])

    assert single.single() == 3.0
    assert np.allclose(columns.values, [[3], [7]])
    assert np.allclose(records.values, [[3], [7]])


def test_predictor_features_select_and_order_a_subset_of_named_input():
    predictor = Predictor(DifferenceRegressor(), features=["x2", "x1"])

    result = predictor.predict(
        {"unused": [100, 200], "x1": [1, 3], "x2": [2, 4]}
    )

    assert np.allclose(result.values, [[1], [1]])


def test_predictor_features_ignore_extras_but_require_declared_features():
    predictor = Predictor(SumRegressor(), features=["x1", "required"])

    with pytest.raises(MissingFeatureError, match="required"):
        predictor.predict({"x1": 1, "unused": 2})


def test_explicit_input_schema_remains_strict_by_default():
    predictor = Predictor(SumRegressor(), schema=InputSchema.from_names(["x1"]))

    with pytest.raises(UnexpectedFeatureError, match="unused"):
        predictor.predict({"x1": 1, "unused": 2})


def test_ensemble_members_select_different_feature_subsets():
    common_input = {
        "a": [1.0, 2.0],
        "b": [10.0, 20.0],
        "c": [100.0, 200.0],
        "d": [1000.0, 2000.0],
    }
    ensemble = EnsemblePredictor(
        [
            Predictor(SumRegressor(), features=["a", "b"]),
            Predictor(SumRegressor(), features=["c", "d"]),
        ],
        aggregation="weighted_mean",
        aggregation_weights=[0.25, 0.75],
    )

    result = ensemble.predict(common_input)

    assert np.allclose(result.values[:, 0], [827.75, 1655.5])


def test_predictor_accepts_list_model_output():
    predictor = Predictor(ListRegressor())

    result = predictor.predict([[1, 2], [3, 4]])

    assert np.allclose(result.values, [[1], [3]])


def test_predictor_applies_standardizer():
    predictor = Predictor(
        SumRegressor(),
        features=["x1", "x2"],
        preprocessor=Standardizer([1, 10], [2, 5]),
    )

    result = predictor.predict({"x1": 3, "x2": 20})

    assert result.single() == 3.0


def test_classifier_predict_and_predict_proba_have_metadata():
    predictor = Predictor(BinaryClassifier(), features=["score"])

    labels = predictor.predict({"score": 0.8})
    probabilities = predictor.predict_proba({"score": [0.2, 0.8]})

    assert labels.single() == "yes"
    assert labels.output_kind == "labels"
    assert probabilities.values.shape == (2, 2)
    assert probabilities.output_kind == "probabilities"
    assert probabilities.classes.tolist() == ["no", "yes"]


def test_configured_probability_output_uses_model_predict():
    predictor = Predictor(
        ForwardProbabilityClassifier(),
        features=["score"],
        task="classification",
        output_kind="probabilities",
    )

    result = predictor.predict_proba({"score": 0.75})

    assert np.allclose(result.values, [[0.25, 0.75]])


def test_named_input_requires_features_or_schema():
    with pytest.raises(ConfigurationError, match="Named input"):
        Predictor(SumRegressor()).predict({"x1": 1})


def test_preprocessor_errors_are_wrapped():
    def fail(_):
        raise RuntimeError("broken")

    with pytest.raises(PreprocessingError, match="broken"):
        Predictor(SumRegressor(), preprocessor=fail).predict([[1, 2]])
