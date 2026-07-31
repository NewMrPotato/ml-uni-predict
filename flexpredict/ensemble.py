"""Safe aggregation of heterogeneous predictors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import (
    ConfigurationError,
    EnsembleCompatibilityError,
    OutputValidationError,
)
from .result import PredictionResult

Aggregation = str | Callable[[np.ndarray], np.ndarray]


class EnsemblePredictor:
    """Combine compatible predictions produced by multiple predictors."""

    def __init__(
        self,
        predictors: Sequence[Any],
        *,
        aggregation: Aggregation = "mean",
        aggregation_weights: Sequence[float] | np.ndarray | str | Path | None = None,
        name: str = "ensemble",
    ) -> None:
        self.predictors = tuple(predictors)
        if not self.predictors:
            raise ConfigurationError("An ensemble requires at least one predictor.")
        for predictor in self.predictors:
            if not callable(getattr(predictor, "predict", None)):
                raise ConfigurationError("Every ensemble member must define predict(data).")
        self.aggregation = aggregation
        self.name = name
        self.weights = self._prepare_weights(aggregation_weights)
        if isinstance(aggregation, str):
            valid = {"mean", "weighted_mean", "median", "min", "max", "voting"}
            if aggregation not in valid:
                raise ConfigurationError(
                    f"Unknown aggregation {aggregation!r}; expected one of {sorted(valid)}."
                )
            if aggregation == "weighted_mean" and self.weights is None:
                self.weights = np.full(len(self.predictors), 1.0 / len(self.predictors))
            if aggregation != "weighted_mean" and self.weights is not None:
                raise ConfigurationError(
                    "aggregation_weights can be used only with aggregation='weighted_mean'."
                )
        elif not callable(aggregation):
            raise ConfigurationError("aggregation must be a supported string or callable.")

    def predict(self, data: Any) -> PredictionResult:
        return self._predict_with("predict", data)

    def predict_proba(self, data: Any) -> PredictionResult:
        return self._predict_with("predict_proba", data)

    def _predict_with(self, method_name: str, data: Any) -> PredictionResult:
        results: list[PredictionResult] = []
        for index, predictor in enumerate(self.predictors):
            method = getattr(predictor, method_name, None)
            if not callable(method):
                raise EnsembleCompatibilityError(
                    f"Predictor {index} does not support {method_name}()."
                )
            result = method(data)
            if not isinstance(result, PredictionResult):
                raise EnsembleCompatibilityError(
                    f"Predictor {index} returned {type(result).__name__}, "
                    "expected PredictionResult."
                )
            results.append(result)

        aligned = self._validate_and_align(results)
        stacked = np.stack([result.values for result in aligned], axis=0)
        values = self._aggregate(stacked, aligned[0])
        return PredictionResult(
            values=values,
            task=aligned[0].task,
            output_kind=aligned[0].output_kind,
            classes=aligned[0].classes,
            is_single=aligned[0].is_single,
            model_name=self.name,
        )

    def _validate_and_align(self, results: list[PredictionResult]) -> list[PredictionResult]:
        reference = results[0]
        aligned = [reference]
        for index, result in enumerate(results[1:], start=1):
            if result.task != reference.task:
                raise EnsembleCompatibilityError(
                    f"Predictor {index} has task {result.task!r}, expected {reference.task!r}."
                )
            if result.output_kind != reference.output_kind:
                raise EnsembleCompatibilityError(
                    f"Predictor {index} returned {result.output_kind!r}, "
                    f"expected {reference.output_kind!r}."
                )
            if result.values.shape != reference.values.shape:
                raise EnsembleCompatibilityError(
                    f"Predictor {index} returned shape {result.values.shape}, "
                    f"expected {reference.values.shape}."
                )
            aligned.append(self._align_classes(result, reference, index))
        return aligned

    def _align_classes(
        self,
        result: PredictionResult,
        reference: PredictionResult,
        index: int,
    ) -> PredictionResult:
        if reference.classes is None and result.classes is None:
            return result
        if reference.classes is None or result.classes is None:
            raise EnsembleCompatibilityError(
                f"Predictor {index} has class metadata incompatible with the reference model."
            )
        if np.array_equal(reference.classes, result.classes):
            return result
        if set(reference.classes.tolist()) != set(result.classes.tolist()):
            raise EnsembleCompatibilityError(
                f"Predictor {index} uses a different set of classes."
            )
        if result.output_kind not in {"probabilities", "logits"}:
            raise EnsembleCompatibilityError(
                "Class order can be aligned only for probability or logit outputs."
            )
        positions = {value: pos for pos, value in enumerate(result.classes.tolist())}
        order = [positions[value] for value in reference.classes.tolist()]
        return PredictionResult(
            values=result.values[:, order],
            task=result.task,
            output_kind=result.output_kind,
            classes=reference.classes,
            is_single=result.is_single,
            model_name=result.model_name,
        )

    def _aggregate(self, stacked: np.ndarray, reference: PredictionResult) -> np.ndarray:
        if callable(self.aggregation):
            try:
                values = np.asarray(self.aggregation(stacked))
            except Exception as exc:
                raise OutputValidationError(f"Custom aggregation failed: {exc}") from exc
        elif self.aggregation == "mean":
            self._require_numeric(stacked)
            values = np.mean(stacked, axis=0)
        elif self.aggregation == "weighted_mean":
            self._require_numeric(stacked)
            values = np.average(stacked, axis=0, weights=self.weights)
        elif self.aggregation == "median":
            self._require_numeric(stacked)
            values = np.median(stacked, axis=0)
        elif self.aggregation == "min":
            self._require_numeric(stacked)
            values = np.min(stacked, axis=0)
        elif self.aggregation == "max":
            self._require_numeric(stacked)
            values = np.max(stacked, axis=0)
        elif self.aggregation == "voting":
            if reference.output_kind != "labels" or reference.values.shape[1] != 1:
                raise EnsembleCompatibilityError(
                    "Voting requires single-output class labels."
                )
            values = _majority_vote(stacked[:, :, 0]).reshape(-1, 1)
        else:  # pragma: no cover - validated in __init__
            raise ConfigurationError(f"Unsupported aggregation: {self.aggregation!r}.")

        if values.shape != reference.values.shape:
            raise OutputValidationError(
                f"Aggregation returned shape {values.shape}, expected {reference.values.shape}."
            )
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise OutputValidationError("Aggregation returned NaN or infinite values.")
        return values

    def _prepare_weights(
        self,
        weights: Sequence[float] | np.ndarray | str | Path | None,
    ) -> np.ndarray | None:
        if weights is None:
            return None
        if isinstance(weights, (str, Path)):
            path = Path(weights)
            if path.suffix.lower() != ".npy":
                raise ConfigurationError("Aggregation weights files must use the .npy format.")
            if not path.is_file():
                raise ConfigurationError(f"Aggregation weights file does not exist: {path}.")
            try:
                weights = np.load(path, allow_pickle=False)
            except Exception as exc:
                raise ConfigurationError(
                    f"Could not load aggregation weights from {path}: {exc}"
                ) from exc
        array = np.asarray(weights, dtype=float)
        if array.ndim != 1 or len(array) != len(self.predictors):
            raise ConfigurationError(
                f"Expected {len(self.predictors)} aggregation weights, "
                f"received shape {array.shape}."
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ConfigurationError(
                "Aggregation weights must be finite and non-negative."
            )
        total = float(array.sum())
        if total <= 0:
            raise ConfigurationError("Aggregation weights must have a positive sum.")
        return array / total

    @staticmethod
    def _require_numeric(stacked: np.ndarray) -> None:
        if not np.issubdtype(stacked.dtype, np.number):
            raise EnsembleCompatibilityError(
                "This aggregation strategy requires numeric model outputs."
            )


def _majority_vote(labels: np.ndarray) -> np.ndarray:
    output: list[Any] = []
    for sample in labels.T:
        values, counts = np.unique(sample, return_counts=True)
        output.append(values[np.argmax(counts)])
    return np.asarray(output)
