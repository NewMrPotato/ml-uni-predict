import numpy as np
import pytest

from flexpredict import (
    ConfigurationError,
    Predictor,
    PreprocessingError,
    Standardizer,
)


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


class ListRegressor:
    def predict(self, values):
        return [float(row[0]) for row in values]


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
