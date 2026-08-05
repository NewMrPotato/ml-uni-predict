"""The main model predictor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from .engines import (
    InferenceEngine,
    create_engine,
    infer_model_metadata,
    normalize_output,
)
from .exceptions import ConfigurationError
from .loading import LoaderName
from .preprocessing import apply_preprocessor
from .result import OutputKind, PredictionResult, Task
from .schema import InputSchema, process_untyped_array


class Predictor:
    """Compose input handling, preprocessing and model inference.

    Args:
        model: Framework model or custom object executed by the inference engine.
        features: Ordered feature names to select from named input. Extra fields are ignored.
        schema: Explicit input contract. Mutually exclusive with ``features``.
        preprocessor: Callable or object providing ``transform(X)``.
        task: Prediction semantics. Inferred for sklearn-compatible models when omitted.
        output_kind: Meaning of regular model output. Defaults to labels for inferred
            classifiers and values otherwise.
        engine: Registered engine name, an engine instance, or ``"auto"``.
        engine_options: Keyword arguments passed to the selected engine factory.
        output_selector: Key, index or callable used to select one native model output.
        name: Human-readable model name used in results and ensemble failures.
    """

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
            InputSchema.from_names(features, dtype=None, extra_fields="ignore")
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

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        loader: LoaderName = "auto",
        loader_options: dict[str, Any] | None = None,
        **predictor_options: Any,
    ) -> Predictor:
        """Load a complete model artifact and create a predictor.

        ``loader_options`` are forwarded to the artifact loader. Remaining keyword
        arguments are forwarded to the predictor constructor.
        """

        from .loading import load_model

        model = load_model(path, loader=loader, loader_options=loader_options)
        return cls(model, **predictor_options)

    @classmethod
    def from_torch_weights(
        cls,
        weights_path: str | Path,
        model_factory: Callable[[], Any],
        *,
        strict: bool = True,
        state_dict_key: str | None = None,
        load_options: dict[str, Any] | None = None,
        **predictor_options: Any,
    ) -> Predictor:
        """Create a PyTorch model from a factory and apply a state dict.

        The checkpoint defaults to weights-only loading. ``state_dict_key`` selects a nested
        checkpoint entry; a top-level ``state_dict`` entry is recognized automatically.
        """

        from .loading import load_torch_state_dict

        engine_options = dict(predictor_options.get("engine_options") or {})
        model = load_torch_state_dict(
            model_factory,
            weights_path,
            strict=strict,
            state_dict_key=state_dict_key,
            map_location=engine_options.get("device", "cpu"),
            load_options=load_options,
        )
        predictor_options["engine"] = "torch"
        return cls(model, **predictor_options)

    def predict(self, data: Any) -> PredictionResult:
        """Validate input, run regular inference and return a canonical result."""

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
        """Return classification probabilities for one sample or a batch."""

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
        if self.preprocessor is None:
            values = batch.values
            if not self._engine.preserves_dataframe:
                values = np.asarray(values)
        else:
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
