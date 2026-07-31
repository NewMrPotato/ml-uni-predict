import numpy as np
import pytest

from flexpredict import Predictor

sklearn = pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")
pd = pytest.importorskip("pandas")
linear_model = pytest.importorskip("sklearn.linear_model")
LinearRegression = linear_model.LinearRegression
LogisticRegression = linear_model.LogisticRegression

pytestmark = pytest.mark.sklearn


def test_sklearn_regression_and_dataframe_input():
    model = LinearRegression().fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        np.array([0.0, 2.0, 4.0]),
    )
    predictor = Predictor(model, features=["x1", "x2"])

    result = predictor.predict(pd.DataFrame([{"x1": 3.0, "x2": 3.0}]))

    assert result.task == "regression"
    assert result.values.shape == (1, 1)
    assert result.single() == pytest.approx(6.0)


def test_sklearn_classification_probabilities():
    model = LogisticRegression().fit(
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        np.array([0, 0, 1, 1]),
    )
    predictor = Predictor(model, features=["score"])

    result = predictor.predict_proba({"score": [-1.0, 1.0]})

    assert result.task == "classification"
    assert result.output_kind == "probabilities"
    assert result.values.shape == (2, 2)
    assert result.classes.tolist() == [0, 1]


def test_joblib_artifact_round_trip(tmp_path):
    model = LinearRegression().fit(np.array([[0.0], [1.0]]), np.array([0.0, 2.0]))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    predictor = Predictor.from_file(path, features=["x"])

    assert predictor.predict({"x": 2.0}).single() == pytest.approx(4.0)
