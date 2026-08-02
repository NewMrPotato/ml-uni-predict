import numpy as np
import pytest

from flexpredict import EnsemblePredictor, Predictor

sklearn = pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")
pd = pytest.importorskip("pandas")
linear_model = pytest.importorskip("sklearn.linear_model")
compose = pytest.importorskip("sklearn.compose")
pipeline = pytest.importorskip("sklearn.pipeline")
preprocessing = pytest.importorskip("sklearn.preprocessing")
LinearRegression = linear_model.LinearRegression
LogisticRegression = linear_model.LogisticRegression
Ridge = linear_model.Ridge
ColumnTransformer = compose.ColumnTransformer
make_column_selector = compose.make_column_selector
Pipeline = pipeline.Pipeline
OneHotEncoder = preprocessing.OneHotEncoder
StandardScaler = preprocessing.StandardScaler

pytestmark = pytest.mark.sklearn


def test_sklearn_regression_and_dataframe_input():
    model = LinearRegression().fit(
        pd.DataFrame(
            {"x1": [0.0, 1.0, 2.0], "x2": [0.0, 1.0, 2.0]}
        ),
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


def test_sklearn_ensemble_members_can_use_different_feature_orders():
    values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    targets = 2.0 * values[:, 0] + 3.0 * values[:, 1]
    direct = LinearRegression().fit(values, targets)
    reversed_order = LinearRegression().fit(values[:, ::-1], targets)
    ensemble = EnsemblePredictor(
        [
            Predictor(direct, features=["x1", "x2"], name="direct"),
            Predictor(reversed_order, features=["x2", "x1"], name="reversed"),
        ]
    )

    result = ensemble.predict({"x1": 4.0, "x2": 5.0})

    assert result.single() == pytest.approx(23.0)


def test_sklearn_column_transformer_preserves_dataframe_semantics():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [20, 30, 50, 60],
            "city": pd.Series(["A", "B", "A", "B"], dtype="category"),
        }
    )
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), ["age", "income"]),
                        (
                            "category",
                            OneHotEncoder(handle_unknown="ignore"),
                            make_column_selector(dtype_include="category"),
                        ),
                    ]
                ),
            ),
            ("model", Ridge()),
        ]
    ).fit(frame, targets)
    expected = model.predict(frame)

    direct = Predictor(model).predict(frame)
    selected = Predictor(
        model,
        features=["age", "income", "city"],
    ).predict(frame.assign(unused=999))

    assert np.allclose(direct.values[:, 0], expected)
    assert np.allclose(selected.values[:, 0], expected)
