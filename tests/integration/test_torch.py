import numpy as np
import pytest

from flexpredict import EnsemblePredictor, Predictor

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)

    def forward(self, values):
        return self.linear(values)


def make_model():
    model = LinearModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[2.0, 3.0]]))
    return model


def test_torch_predictor_auto_detection_and_batch():
    predictor = Predictor(
        make_model(),
        features=["x1", "x2"],
        task="regression",
        engine_options={"device": "cpu", "dtype": "float32"},
    )

    result = predictor.predict({"x1": [1.0, 2.0], "x2": [2.0, 1.0]})

    assert result.values.shape == (2, 1)
    assert np.allclose(result.values, [[8.0], [7.0]])


def test_torch_state_dict_round_trip(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save(make_model().state_dict(), path)

    predictor = Predictor.from_torch_weights(
        path,
        LinearModel,
        features=["x1", "x2"],
        task="regression",
    )

    assert predictor.predict({"x1": 1.0, "x2": 2.0}).single() == pytest.approx(8.0)


def test_torch_and_generic_models_form_a_heterogeneous_ensemble():
    class ReversedNumpyModel:
        _estimator_type = "regressor"

        def predict(self, values):
            array = np.asarray(values)
            return 3.0 * array[:, 0] + 2.0 * array[:, 1]

    ensemble = EnsemblePredictor(
        [
            Predictor(make_model(), features=["x1", "x2"], task="regression"),
            Predictor(ReversedNumpyModel(), features=["x2", "x1"]),
        ],
        aggregation="weighted_mean",
        aggregation_weights=[0.25, 0.75],
    )

    result = ensemble.predict({"x1": [1.0, 2.0], "x2": [2.0, 1.0]})

    assert np.allclose(result.values, [[8.0], [7.0]])
