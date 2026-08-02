import numpy as np
import pytest

from flexpredict import Predictor

tf = pytest.importorskip("tensorflow")

pytestmark = pytest.mark.tensorflow


def make_model():
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(1, use_bias=False)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.layers[-1].set_weights([np.array([[2.0], [3.0]], dtype=np.float32)])
    return model


def test_tensorflow_predictor_auto_detection():
    predictor = Predictor(
        make_model(),
        features=["x1", "x2"],
        task="regression",
    )

    result = predictor.predict({"x1": [1.0, 2.0], "x2": [2.0, 1.0]})

    assert result.values.shape == (2, 1)
    assert np.allclose(result.values, [[8.0], [7.0]])


def test_keras_artifact_round_trip(tmp_path):
    path = tmp_path / "model.keras"
    make_model().save(path)

    predictor = Predictor.from_file(
        path,
        features=["x1", "x2"],
        task="regression",
        loader_options={"compile": False},
    )

    assert predictor.predict({"x1": 1.0, "x2": 2.0}).single() == pytest.approx(8.0)
