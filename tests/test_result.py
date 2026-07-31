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


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_prediction_result_rejects_non_finite_values(value):
    with pytest.raises(OutputValidationError, match="finite"):
        PredictionResult(np.array([[value]]))


def test_prediction_result_validates_single_metadata():
    with pytest.raises(OutputValidationError, match="one sample"):
        PredictionResult(np.array([[1.0], [2.0]]), is_single=True)


def test_prediction_result_validates_probability_contract():
    with pytest.raises(OutputValidationError, match="numeric"):
        PredictionResult(
            np.array([["low", "high"]]), output_kind="probabilities"
        )
    with pytest.raises(OutputValidationError, match="between 0 and 1"):
        PredictionResult(np.array([[-0.1, 1.1]]), output_kind="probabilities")


def test_prediction_result_requires_unique_non_empty_classes():
    with pytest.raises(OutputValidationError, match="empty"):
        PredictionResult(np.array([[1.0]]), classes=np.array([]))
    with pytest.raises(OutputValidationError, match="unique"):
        PredictionResult(np.array([[1.0]]), classes=np.array(["same", "same"]))
    with pytest.raises(OutputValidationError, match="finite"):
        PredictionResult(np.array([[1.0]]), classes=np.array([np.nan]))
