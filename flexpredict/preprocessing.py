"""Built-in preprocessing and preprocessor execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import ConfigurationError, PreprocessingError


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def __init__(self, mean: Any, std: Any) -> None:
        mean_array = np.asarray(mean, dtype=float)
        std_array = np.asarray(std, dtype=float)
        if mean_array.ndim != 1 or std_array.ndim != 1:
            raise ConfigurationError("Standardizer mean and std must be one-dimensional.")
        if mean_array.shape != std_array.shape or mean_array.size == 0:
            raise ConfigurationError("Standardizer mean and std must have equal non-zero sizes.")
        if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
            raise ConfigurationError("Standardizer parameters must contain only finite values.")
        if np.any(std_array <= 0):
            raise ConfigurationError("Standardizer std values must be greater than zero.")
        object.__setattr__(self, "mean", mean_array)
        object.__setattr__(self, "std", std_array)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise PreprocessingError(
                f"Standardizer expects {self.mean.size} features, received shape {array.shape}."
            )
        return (array - self.mean) / self.std


def apply_preprocessor(preprocessor: Any, values: Any) -> np.ndarray:
    if preprocessor is None:
        return np.asarray(values)
    try:
        if hasattr(preprocessor, "transform") and callable(preprocessor.transform):
            transformed = preprocessor.transform(values)
        elif callable(preprocessor):
            transformed = preprocessor(values)
        else:
            raise TypeError("preprocessor must be callable or define transform(X).")
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    array = np.asarray(transformed)
    if array.ndim != 2:
        raise PreprocessingError(
            f"Preprocessor must return a 2D batch, received shape {array.shape}."
        )
    if array.shape[0] != values.shape[0] or array.shape[1] == 0:
        raise PreprocessingError(
            "Preprocessor must preserve the number of samples and return at least one feature."
        )
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise PreprocessingError("Preprocessor returned NaN or infinite values.")
    return array
