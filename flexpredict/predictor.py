"""The main model predictor."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .engines import (
    InferenceEngine,
    create_engine,
    infer_model_metadata,
    normalize_output,
)
from .exceptions import ConfigurationError
from .preprocessing import apply_preprocessor
from .result import OutputKind, PredictionResult, Task
from .schema import InputSchema, process_untyped_array


class Predictor:
    """Compose input handling, preprocessing and model inference."""

    def __init__(
        self,
        model: Any,
        *,
        features: list[str] | tuple[str, ...] | None = None,
        schema: InputSchema | None = None,
        preprocessor: Any = None,
        task: Task | None = None,
        output_kind: OutputKind | None = None,
        engine: str | InferenceEngine = "auto",
        engine_options: dict[str, Any] | None = None,
        output_selector: Any = None,
        name: str | None = None,
    ) -> None:
        if features is not None and schema is not None:
            raise ConfigurationError("Pass either features or schema, not both.")
        self.model = model
        self.schema = (
            InputSchema.from_names(features, extra_fields="forbid")
            if features is not None
            else schema
        )
        self.preprocessor = preprocessor
        self.name = name or type(model).__name__
        self.engine_options = dict(engine_options or {})
        self._engine = create_engine(
            model,
            engine=engine,
            engine_options=self.engine_options,
            output_selector=output_selector,
        )

        inferred_task, inferred_classes = infer_model_metadata(model)
        self.task = cast(Task, task or inferred_task)
        self.classes = inferred_classes
        if output_kind is None:
            self.output_kind: OutputKind = (
                "labels" if self.task == "classification" else "values"
            )
        else:
            self.output_kind = output_kind
        self._validate_semantics()

    def predict(self, data: Any) -> PredictionResult:
        batch = self._prepare(data)
        raw = self._engine.predict(batch.values)
        values = normalize_output(raw, n_samples=batch.n_samples)
        return PredictionResult(
            values=values,
            task=self.task,
            output_kind=self.output_kind,
            classes=self.classes,
            is_single=batch.is_single,
            model_name=self.name,
        )

    def predict_proba(self, data: Any) -> PredictionResult:
        if self.task not in {"classification", "unknown"}:
            raise ConfigurationError("predict_proba() is valid only for classification models.")
        batch = self._prepare(data)
        raw = (
            self._engine.predict(batch.values)
            if self.output_kind == "probabilities"
            else self._engine.predict_proba(batch.values)
        )
        values = normalize_output(raw, n_samples=batch.n_samples)
        return PredictionResult(
            values=values,
            task="classification",
            output_kind="probabilities",
            classes=self.classes,
            is_single=batch.is_single,
            model_name=self.name,
        )

    def _prepare(self, data: Any) -> Any:
        batch = self.schema.process(data) if self.schema else process_untyped_array(data)
        values = apply_preprocessor(self.preprocessor, batch.values)
        return type(batch)(values, batch.feature_names, batch.is_single)

    def _validate_semantics(self) -> None:
        if self.task not in {"regression", "classification", "unknown"}:
            raise ConfigurationError(f"Unknown task: {self.task!r}.")
        if self.output_kind not in {"values", "labels", "probabilities", "logits"}:
            raise ConfigurationError(f"Unknown output_kind: {self.output_kind!r}.")
        if self.task == "regression" and self.output_kind in {
            "labels",
            "probabilities",
            "logits",
        }:
            raise ConfigurationError(
                f"Regression predictors cannot use output_kind={self.output_kind!r}."
            )
