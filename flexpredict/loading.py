"""Explicit, lazy model-artifact loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from .exceptions import ConfigurationError, EngineNotAvailableError

LoaderName = Literal["auto", "joblib", "tensorflow", "torch_model"]


def load_model(
    path: str | Path,
    *,
    loader: LoaderName = "auto",
    loader_options: dict[str, Any] | None = None,
) -> Any:
    """Load a complete model artifact using an explicit or inferred loader.

    PyTorch ``.pt`` and ``.pth`` files are intentionally not auto-detected because
    they may contain either a complete model, a state dict or an application-specific
    checkpoint. Use ``loader='torch_model'`` only for trusted complete-model files, or
    :func:`load_torch_state_dict` for weights-only checkpoints.
    """

    artifact = _validate_file(path)
    selected = _infer_loader(artifact) if loader == "auto" else loader
    options = dict(loader_options or {})

    if selected == "joblib":
        joblib = _import_joblib()
        try:
            return joblib.load(artifact, **options)
        except Exception as exc:
            raise ConfigurationError(f"Could not load joblib model {artifact}: {exc}") from exc

    if selected == "tensorflow":
        tf = _import_tensorflow()
        try:
            return tf.keras.models.load_model(artifact, **options)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not load TensorFlow/Keras model {artifact}: {exc}"
            ) from exc

    if selected == "torch_model":
        torch = _import_torch()
        options.setdefault("weights_only", False)
        try:
            return torch.load(artifact, **options)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not load complete PyTorch model {artifact}: {exc}"
            ) from exc

    raise ConfigurationError(f"Unknown model loader: {selected!r}.")


def load_torch_state_dict(
    model_factory: Callable[[], Any],
    weights_path: str | Path,
    *,
    strict: bool = True,
    state_dict_key: str | None = None,
    map_location: Any = "cpu",
    load_options: dict[str, Any] | None = None,
) -> Any:
    """Create a PyTorch model and load a weights-only checkpoint into it."""

    if not callable(model_factory):
        raise ConfigurationError("model_factory must be callable.")
    artifact = _validate_file(weights_path)
    torch = _import_torch()
    options = dict(load_options or {})
    options.setdefault("weights_only", True)
    options.setdefault("map_location", map_location)
    try:
        checkpoint = torch.load(artifact, **options)
    except Exception as exc:
        raise ConfigurationError(
            f"Could not load PyTorch state dict {artifact}: {exc}"
        ) from exc

    if state_dict_key is not None:
        try:
            checkpoint = checkpoint[state_dict_key]
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                f"Checkpoint {artifact} has no state dict key {state_dict_key!r}."
            ) from exc
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        model = model_factory()
    except Exception as exc:
        raise ConfigurationError(f"model_factory failed: {exc}") from exc
    method = getattr(model, "load_state_dict", None)
    if not callable(method):
        raise ConfigurationError("model_factory must return an object with load_state_dict().")
    try:
        method(checkpoint, strict=strict)
    except Exception as exc:
        raise ConfigurationError(f"Could not apply PyTorch state dict: {exc}") from exc
    return model


def _validate_file(path: str | Path) -> Path:
    artifact = Path(path)
    if not artifact.is_file():
        raise ConfigurationError(f"Model artifact does not exist or is not a file: {artifact}.")
    return artifact


def _infer_loader(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".joblib", ".pkl", ".pickle"}:
        return "joblib"
    if suffix in {".keras", ".h5", ".hdf5"}:
        return "tensorflow"
    if suffix in {".pt", ".pth"}:
        raise ConfigurationError(
            "PyTorch files are ambiguous. Use Predictor.from_torch_weights(...) for a "
            "state dict or Predictor.from_file(..., loader='torch_model') for a trusted "
            "complete-model artifact."
        )
    raise ConfigurationError(
        f"Cannot infer a model loader from extension {suffix!r}; pass loader=... explicitly."
    )


def _import_joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise EngineNotAvailableError(
            "Joblib model loading requires: pip install flexpredict[sklearn]"
        ) from exc
    return joblib


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise EngineNotAvailableError(
            "PyTorch model loading requires: pip install flexpredict[torch]"
        ) from exc
    return torch


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise EngineNotAvailableError(
            "TensorFlow model loading requires: pip install flexpredict[tensorflow]"
        ) from exc
    return tf

