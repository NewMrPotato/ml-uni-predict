"""Inference engines with lazy optional-framework imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from .exceptions import (
    ConfigurationError,
    EngineNotAvailableError,
    InferenceError,
    OutputValidationError,
    UnsupportedOutputError,
)


class InferenceEngine(ABC):
    """Framework adapter used by :class:`flexpredict.Predictor`."""

    preserves_dataframe = False

    def __init__(self, model: Any, *, output_selector: Any = None) -> None:
        self.model = model
        self.output_selector = output_selector

    @abstractmethod
    def predict(self, values: np.ndarray) -> Any:
        """Run the model and return its native prediction output."""

    def predict_proba(self, values: np.ndarray) -> Any:
        method = getattr(self.model, "predict_proba", None)
        if not callable(method):
            raise UnsupportedOutputError(
                f"{type(self.model).__name__} does not provide predict_proba()."
            )
        return self._call(method, values)

    def _call(self, method: Callable[[Any], Any], values: Any) -> Any:
        try:
            return _select_output(method(values), self.output_selector)
        except (UnsupportedOutputError, OutputValidationError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"{type(self.model).__name__} failed during inference: {exc}"
            ) from exc


class GenericEngine(InferenceEngine):
    preserves_dataframe = True

    def predict(self, values: np.ndarray) -> Any:
        method = getattr(self.model, "predict", None)
        if not callable(method):
            raise ConfigurationError(
                f"{type(self.model).__name__} must define a callable predict(X) method."
            )
        return self._call(method, values)


class TorchEngine(InferenceEngine):
    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        output_selector: Any = None,
    ) -> None:
        super().__init__(model, output_selector=output_selector)
        torch = _import_torch()
        if not isinstance(model, torch.nn.Module):
            raise ConfigurationError("TorchEngine requires an instance of torch.nn.Module.")
        try:
            self.device = torch.device(device)
            self.dtype = getattr(torch, dtype)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise ConfigurationError(
                f"Invalid PyTorch device or dtype: device={device!r}, dtype={dtype!r}."
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    def predict(self, values: np.ndarray) -> Any:
        torch = _import_torch()
        try:
            tensor = torch.as_tensor(
                np.asarray(values), dtype=self.dtype, device=self.device
            )
            with torch.inference_mode():
                output = self.model(tensor)
            return _select_output(output, self.output_selector)
        except (OutputValidationError, UnsupportedOutputError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"{type(self.model).__name__} failed during PyTorch inference: {exc}"
            ) from exc

    def predict_proba(self, values: np.ndarray) -> Any:
        method = getattr(self.model, "predict_proba", None)
        if callable(method):
            return self._call(method, values)
        raise UnsupportedOutputError(
            "PyTorch models require an explicit probability-producing adapter or predict_proba()."
        )


class TensorFlowEngine(InferenceEngine):
    def __init__(self, model: Any, *, output_selector: Any = None) -> None:
        super().__init__(model, output_selector=output_selector)
        tf = _import_tensorflow()
        if not isinstance(model, tf.keras.Model):
            raise ConfigurationError("TensorFlowEngine requires an instance of tf.keras.Model.")

    def predict(self, values: np.ndarray) -> Any:
        try:
            return _select_output(
                self.model.predict(np.asarray(values), verbose=0),
                self.output_selector,
            )
        except (OutputValidationError, UnsupportedOutputError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"{type(self.model).__name__} failed during TensorFlow inference: {exc}"
            ) from exc


EngineFactory = Callable[..., InferenceEngine]
EngineDetector = Callable[[Any], bool]
_ENGINE_FACTORIES: dict[str, EngineFactory] = {
    "generic": GenericEngine,
    "sklearn": GenericEngine,
    "torch": TorchEngine,
    "tensorflow": TensorFlowEngine,
}
_CUSTOM_DETECTORS: list[tuple[int, str, EngineDetector]] = []


def register_engine(
    name: str,
    factory: EngineFactory,
    *,
    detector: EngineDetector | None = None,
    priority: int = 0,
    replace: bool = False,
) -> None:
    """Register an engine factory and an optional auto-detection function.

    Higher-priority custom detectors run first and before built-in framework detection.
    Existing engine names require ``replace=True``.
    """

    if not name or not callable(factory):
        raise ConfigurationError("Engine name and callable factory are required.")
    if name in _ENGINE_FACTORIES and not replace:
        raise ConfigurationError(f"Engine {name!r} is already registered.")
    _ENGINE_FACTORIES[name] = factory
    if detector is not None:
        if not callable(detector):
            raise ConfigurationError("Engine detector must be callable.")
        _CUSTOM_DETECTORS.append((priority, name, detector))
        _CUSTOM_DETECTORS.sort(key=lambda item: item[0], reverse=True)


def create_engine(
    model: Any,
    *,
    engine: str | InferenceEngine = "auto",
    engine_options: dict[str, Any] | None = None,
    output_selector: Any = None,
) -> InferenceEngine:
    if isinstance(engine, InferenceEngine):
        if engine_options:
            raise ConfigurationError(
                "engine_options cannot be used with an engine instance."
            )
        return engine

    name = _detect_engine(model) if engine == "auto" else engine
    try:
        factory = _ENGINE_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENGINE_FACTORIES))
        raise ConfigurationError(
            f"Unknown engine {name!r}. Available engines: {available}."
        ) from exc
    options = dict(engine_options or {})
    options["output_selector"] = output_selector
    try:
        return factory(model, **options)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid options for engine {name!r}: {exc}") from exc


def infer_model_metadata(model: Any) -> tuple[str, np.ndarray | None]:
    estimator_type = _infer_estimator_type(model)
    classes = getattr(model, "classes_", None)
    class_array = None if classes is None else np.asarray(classes)
    if estimator_type == "classifier":
        return "classification", class_array
    if estimator_type == "regressor":
        return "regression", None
    return "unknown", class_array


def _infer_estimator_type(model: Any) -> str | None:
    """Read modern sklearn tags without making sklearn a base dependency."""

    tags_method = getattr(model, "__sklearn_tags__", None)
    if callable(tags_method):
        try:
            estimator_type = getattr(tags_method(), "estimator_type", None)
            if estimator_type in {"classifier", "regressor"}:
                return estimator_type
        except Exception:
            # Third-party sklearn-compatible estimators occasionally expose partial
            # tag implementations. The legacy attribute remains a useful fallback.
            pass
    estimator_type = getattr(model, "_estimator_type", None)
    return estimator_type if estimator_type in {"classifier", "regressor"} else None


def normalize_output(raw: Any, *, n_samples: int) -> np.ndarray:
    """Convert a native model output to ``(n_samples, n_outputs)``."""

    value = _to_numpy(raw)
    if value.ndim == 0:
        if n_samples != 1:
            raise OutputValidationError(
                "A scalar model output is valid only for a single input sample."
            )
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        if n_samples == 1:
            value = value.reshape(1, -1)
        elif len(value) == n_samples:
            value = value.reshape(n_samples, 1)
        else:
            raise OutputValidationError(
                f"One-dimensional output has length {len(value)}, expected {n_samples}."
            )
    elif value.ndim == 2:
        if value.shape[0] != n_samples:
            raise OutputValidationError(
                f"Model returned {value.shape[0]} samples, expected {n_samples}."
            )
    else:
        raise OutputValidationError(
            f"Model output must be scalar, 1D or 2D; received shape {value.shape}."
        )
    if value.shape[1] == 0:
        raise OutputValidationError("Model output cannot have zero columns.")
    if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
        raise OutputValidationError("Model output contains NaN or infinite values.")
    return value


def _detect_engine(model: Any) -> str:
    for _, name, detector in _CUSTOM_DETECTORS:
        try:
            if detector(model):
                return name
        except Exception as exc:
            raise ConfigurationError(f"Detector for engine {name!r} failed: {exc}") from exc

    if _belongs_to_framework(model, "torch"):
        return "torch"
    if _belongs_to_framework(model, "tensorflow") or _belongs_to_framework(model, "keras"):
        return "tensorflow"
    if callable(getattr(model, "predict", None)):
        return "generic"
    raise ConfigurationError(
        f"Cannot detect an inference engine for {type(model).__name__}. "
        "Pass engine=... explicitly or register a custom engine."
    )


def _select_output(output: Any, selector: Any) -> Any:
    if selector is None:
        if isinstance(output, dict):
            if len(output) == 1:
                return next(iter(output.values()))
            raise OutputValidationError(
                "The model returned multiple named outputs; configure output_selector."
            )
        return output
    try:
        if callable(selector):
            return selector(output)
        return output[selector]
    except Exception as exc:
        raise OutputValidationError(
            f"output_selector {selector!r} could not select a model output."
        ) from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise OutputValidationError(
            f"Model output of type {type(value).__name__} cannot be converted to NumPy."
        ) from exc


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise EngineNotAvailableError(
            "PyTorch support requires: pip install flexpredict[torch]"
        ) from exc
    return torch


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise EngineNotAvailableError(
            "TensorFlow support requires: pip install flexpredict[tensorflow]"
        ) from exc
    return tf


def _belongs_to_framework(model: Any, framework: str) -> bool:
    return any(
        base.__module__.split(".", 1)[0] == framework
        for base in type(model).__mro__
    )
