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


@pytest.mark.parametrize("output_kind", ["values", "probabilities", "logits"])
def test_numeric_outputs_reject_complex_values(output_kind):
    with pytest.raises(OutputValidationError, match="real numeric"):
        PredictionResult(
            np.array([[1 + 2j]], dtype=np.complex128),
            output_kind=output_kind,
        )


def test_value_outputs_reject_strings_and_object_arrays():
    arbitrary = np.empty((1, 1), dtype=object)
    arbitrary[0, 0] = {"prediction": 1}

    with pytest.raises(OutputValidationError, match="real numeric"):
        PredictionResult(np.array([["broken"]]))
    with pytest.raises(OutputValidationError, match="real numeric"):
        PredictionResult(arbitrary)


def test_classification_labels_accept_supported_scalar_types():
    labels = np.array([["cat"], [1], [True], [2.5]], dtype=object)

    result = PredictionResult(
        labels,
        task="classification",
        output_kind="labels",
    )

    assert result.values.tolist() == [["cat"], [1], [True], [2.5]]


def test_regression_results_cannot_bypass_output_kind_semantics():
    with pytest.raises(OutputValidationError, match="Regression results"):
        PredictionResult(
            np.array([["label"]]),
            task="regression",
            output_kind="labels",
        )


@pytest.mark.parametrize("label", [{"class": 1}, ["nested"], None, 1 + 2j, b"bytes"])
def test_classification_labels_reject_unsupported_objects(label):
    labels = np.empty((1, 1), dtype=object)
    labels[0, 0] = label

    with pytest.raises(OutputValidationError, match="unsupported"):
        PredictionResult(
            labels,
            task="classification",
            output_kind="labels",
        )


def test_prediction_result_requires_unique_non_empty_classes():
    with pytest.raises(OutputValidationError, match="empty"):
        PredictionResult(np.array([[1.0]]), classes=np.array([]))
    with pytest.raises(OutputValidationError, match="unique"):
        PredictionResult(np.array([[1.0]]), classes=np.array(["same", "same"]))
    with pytest.raises(OutputValidationError, match="finite"):
        PredictionResult(np.array([[1.0]]), classes=np.array([np.nan]))
    unsupported = np.empty(1, dtype=object)
    unsupported[0] = {"class": 1}
    with pytest.raises(OutputValidationError, match="unsupported"):
        PredictionResult(np.array([[1.0]]), classes=unsupported)
