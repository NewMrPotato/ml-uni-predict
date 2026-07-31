"""Canonical prediction result returned by FlexPredict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .exceptions import OutputValidationError

Task = Literal["regression", "classification", "unknown"]
OutputKind = Literal["values", "labels", "probabilities", "logits"]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A framework-independent prediction batch.

    ``values`` always has shape ``(n_samples, n_outputs)``. Whether the
    caller supplied one record is represented by ``is_single`` rather than
    by squeezing the array and losing shape information.
    """

    values: np.ndarray
    task: Task = "unknown"
    output_kind: OutputKind = "values"
    classes: np.ndarray | None = None
    is_single: bool = False
    model_name: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise OutputValidationError(
                f"PredictionResult.values must be 2D, received shape {values.shape}."
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise OutputValidationError("PredictionResult.values cannot be empty.")
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise OutputValidationError("PredictionResult.values must contain only finite values.")
        if self.is_single and values.shape[0] != 1:
            raise OutputValidationError(
                "is_single=True requires PredictionResult.values to contain one sample."
            )
        object.__setattr__(self, "values", values)

        if self.task not in {"regression", "classification", "unknown"}:
            raise OutputValidationError(f"Unknown prediction task: {self.task!r}.")
        if self.output_kind not in {"values", "labels", "probabilities", "logits"}:
            raise OutputValidationError(f"Unknown output kind: {self.output_kind!r}.")
        if self.output_kind in {"probabilities", "logits"} and not np.issubdtype(
            values.dtype, np.number
        ):
            raise OutputValidationError(
                f"{self.output_kind.capitalize()} must contain numeric values."
            )
        if self.output_kind == "probabilities" and (
            np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12)
        ):
            raise OutputValidationError("Probabilities must be between 0 and 1.")

        if self.classes is not None:
            classes = np.asarray(self.classes)
            if classes.ndim != 1:
                raise OutputValidationError("classes must be a one-dimensional array.")
            if classes.size == 0:
                raise OutputValidationError("classes cannot be empty.")
            if np.issubdtype(classes.dtype, np.number) and not np.all(
                np.isfinite(classes)
            ):
                raise OutputValidationError("classes must contain only finite values.")
            if _contains_duplicates(classes):
                raise OutputValidationError("classes must contain unique values.")
            if self.output_kind in {"probabilities", "logits"} and len(classes) != values.shape[1]:
                raise OutputValidationError(
                    "The number of classes must match the number of output columns."
                )
            object.__setattr__(self, "classes", classes)

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_outputs(self) -> int:
        return int(self.values.shape[1])

    def single(self) -> Any:
        """Return one prediction as a scalar or a one-dimensional copy."""

        if not self.is_single and self.n_samples != 1:
            raise OutputValidationError(
                f"single() requires one sample, but the result contains {self.n_samples}."
            )
        row = self.values[0]
        if self.n_outputs == 1:
            return row[0].item() if hasattr(row[0], "item") else row[0]
        return row.copy()


def _contains_duplicates(values: np.ndarray) -> bool:
    items = values.tolist()
    return any(
        _labels_equal(value, previous)
        for index, value in enumerate(items)
        for previous in items[:index]
    )


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result) if np.isscalar(result) else bool(np.all(result))
    except (TypeError, ValueError):
        return False
