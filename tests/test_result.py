import numpy as np
import pytest

from flexpredict import OutputValidationError, PredictionResult


def test_prediction_result_has_canonical_shape_and_single_value():
    result = PredictionResult(
        values=np.array([[42.0]]),
        task="regression",
        is_single=True,
    )

    assert result.values.shape == (1, 1)
    assert result.single() == 42.0
    assert result.n_samples == 1
    assert result.n_outputs == 1


def test_prediction_result_returns_multioutput_row():
    result = PredictionResult(np.array([[0.25, 0.75]]), is_single=True)

    assert np.allclose(result.single(), [0.25, 0.75])


def test_prediction_result_rejects_non_2d_values():
    with pytest.raises(OutputValidationError, match="must be 2D"):
        PredictionResult(np.array([1.0, 2.0]))

