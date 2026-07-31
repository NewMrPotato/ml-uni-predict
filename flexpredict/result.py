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
        object.__setattr__(self, "values", values)

        if self.task not in {"regression", "classification", "unknown"}:
            raise OutputValidationError(f"Unknown prediction task: {self.task!r}.")
        if self.output_kind not in {"values", "labels", "probabilities", "logits"}:
            raise OutputValidationError(f"Unknown output kind: {self.output_kind!r}.")

        if self.classes is not None:
            classes = np.asarray(self.classes)
            if classes.ndim != 1:
                raise OutputValidationError("classes must be a one-dimensional array.")
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

