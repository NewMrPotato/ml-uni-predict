"""Schema-aware conversion of supported tabular input formats."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .exceptions import (
    ConfigurationError,
    InputValidationError,
    MissingFeatureError,
    UnexpectedFeatureError,
)


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    dtype: Any = float
    required: bool = True
    nullable: bool = False
    default: Any = MISSING
    validators: tuple[Callable[[Any], bool | None], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("Feature names must be non-empty strings.")
        object.__setattr__(self, "validators", tuple(self.validators))

    def validate(self, value: Any, *, coerce: bool) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise InputValidationError(f"Feature {self.name!r} cannot be null.")

        converted = value
        if coerce and self.dtype is not None:
            try:
                converted = self.dtype(value)
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    f"Feature {self.name!r} cannot be converted to {self.dtype!r}: {value!r}."
                ) from exc
        elif self.dtype is not None:
            try:
                expected = np.dtype(self.dtype)
                actual = np.asarray(value).dtype
            except TypeError as exc:
                raise ConfigurationError(
                    f"Unsupported dtype for feature {self.name!r}: {self.dtype!r}."
                ) from exc
            if not np.can_cast(actual, expected, casting="safe"):
                raise InputValidationError(
                    f"Feature {self.name!r} has dtype {actual}, expected {expected}."
                )

        for validator in self.validators:
            try:
                valid = validator(converted)
            except Exception as exc:
                raise InputValidationError(
                    f"Validator for feature {self.name!r} failed: {exc}"
                ) from exc
            if valid is False:
                raise InputValidationError(
                    f"Feature {self.name!r} failed validation for value {converted!r}."
                )
        return converted


@dataclass(frozen=True, slots=True)
class InputBatch:
    """Validated 2D input, retaining a pandas DataFrame when names carry semantics."""

    values: Any
    feature_names: tuple[str, ...]
    is_single: bool

    def __post_init__(self) -> None:
        values = self.values if _is_pandas_dataframe(self.values) else np.asarray(self.values)
        if values.ndim != 2:
            raise InputValidationError(
                f"Input batch must be 2D, received shape {values.shape}."
            )
        if values.shape[0] == 0:
            raise InputValidationError("Input batch cannot be empty.")
        if values.shape[1] != len(self.feature_names):
            raise InputValidationError(
                f"Input has {values.shape[1]} columns but {len(self.feature_names)} are expected."
            )
        object.__setattr__(self, "values", values)

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class InputSchema:
    features: tuple[FeatureSpec, ...]
    extra_fields: Literal["forbid", "ignore"] = "forbid"
    coerce: bool = True

    def __post_init__(self) -> None:
        features = tuple(self.features)
        if not features:
            raise ConfigurationError("InputSchema must contain at least one feature.")
        names = [feature.name for feature in features]
        if len(names) != len(set(names)):
            raise ConfigurationError("InputSchema feature names must be unique.")
        if self.extra_fields not in {"forbid", "ignore"}:
            raise ConfigurationError("extra_fields must be 'forbid' or 'ignore'.")
        object.__setattr__(self, "features", features)

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        *,
        dtype: Any = float,
        extra_fields: Literal["forbid", "ignore"] = "forbid",
    ) -> InputSchema:
        return cls(
            tuple(FeatureSpec(name=name, dtype=dtype) for name in names),
            extra_fields=extra_fields,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def process(self, data: Any) -> InputBatch:
        if _is_pandas_dataframe(data):
            return self._process_dataframe(data)
        if _is_pandas_object(data):
            data = _pandas_to_supported(data)

        if isinstance(data, Mapping):
            rows, is_single = self._rows_from_mapping(data)
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            rows, is_single = self._rows_from_structured(data)
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, np.ndarray)):
            if len(data) == 0:
                raise InputValidationError("Input data cannot be empty.")
            if all(isinstance(item, Mapping) for item in data):
                rows = [self._validate_mapping(item) for item in data]
                is_single = False
            else:
                return self._process_array(data)
        else:
            return self._process_array(data)

        return InputBatch(
            values=np.asarray(rows),
            feature_names=self.feature_names,
            is_single=is_single,
        )

    # Covered by the optional pandas/sklearn integration suite.
    def _process_dataframe(self, data: Any) -> InputBatch:  # pragma: no cover
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise InputValidationError("Pandas DataFrame input cannot be empty.")
        if data.columns.has_duplicates:
            raise InputValidationError("Pandas DataFrame columns must be unique.")

        columns = set(data.columns.tolist())
        expected = set(self.feature_names)
        extras = columns - expected
        if extras and self.extra_fields == "forbid":
            raise UnexpectedFeatureError(
                f"Unexpected features: {', '.join(sorted(map(str, extras)))}."
            )

        all_declared_columns_present = all(
            name in columns for name in self.feature_names
        )
        can_preserve_dtypes = all_declared_columns_present and (
            not self.coerce or all(feature.dtype is None for feature in self.features)
        )
        if can_preserve_dtypes:
            for feature in self.features:
                for value in data[feature.name].tolist():
                    feature.validate(value, coerce=self.coerce)
            selected = data.loc[:, list(self.feature_names)].copy()
            return InputBatch(selected, self.feature_names, is_single=False)

        rows = [self._validate_mapping(item) for item in data.to_dict(orient="records")]
        validated = type(data)(rows, columns=self.feature_names, index=data.index)
        return InputBatch(validated, self.feature_names, is_single=False)

    def _rows_from_mapping(self, data: Mapping[str, Any]) -> tuple[list[list[Any]], bool]:
        if not data:
            raise InputValidationError("Input mapping cannot be empty.")

        present_values = [data[name] for name in self.feature_names if name in data]
        if not present_values:
            return [self._validate_mapping(data)], True

        sequence_flags = [_is_column(value) for value in present_values]
        if any(sequence_flags) and not all(sequence_flags):
            raise InputValidationError(
                "A mapping cannot mix scalar feature values with batch columns."
            )
        if not any(sequence_flags):
            return [self._validate_mapping(data)], True

        lengths = {len(value) for value in present_values}
        if len(lengths) != 1:
            raise InputValidationError("All feature columns must have the same length.")
        n_samples = lengths.pop()
        if n_samples == 0:
            raise InputValidationError("Input columns cannot be empty.")
        rows = [
            self._validate_mapping(
                {
                    key: (value[index] if _is_column(value) else value)
                    for key, value in data.items()
                }
            )
            for index in range(n_samples)
        ]
        return rows, False

    def _rows_from_structured(self, data: np.ndarray) -> tuple[list[list[Any]], bool]:
        if data.ndim != 1 or len(data) == 0:
            raise InputValidationError("Structured arrays must be non-empty and one-dimensional.")
        rows = [
            self._validate_mapping({name: row[name] for name in data.dtype.names or ()})
            for row in data
        ]
        return rows, len(rows) == 1

    def _validate_mapping(self, item: Mapping[str, Any]) -> list[Any]:
        keys = set(item)
        expected = set(self.feature_names)
        extras = keys - expected
        if extras and self.extra_fields == "forbid":
            raise UnexpectedFeatureError(
                f"Unexpected features: {', '.join(sorted(map(str, extras)))}."
            )

        row: list[Any] = []
        for feature in self.features:
            if feature.name in item:
                value = item[feature.name]
            elif feature.default is not MISSING:
                value = feature.default
            elif feature.required:
                raise MissingFeatureError(f"Required feature {feature.name!r} is missing.")
            else:
                value = None
            row.append(feature.validate(value, coerce=self.coerce))
        return row

    def _process_array(self, data: Any) -> InputBatch:
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(1, -1)
            is_single = True
        elif array.ndim == 2:
            is_single = False
        else:
            raise InputValidationError(
                f"Array input must be 1D or 2D, received shape {array.shape}."
            )
        if array.shape[0] == 0:
            raise InputValidationError("Input array cannot be empty.")
        if array.shape[1] != len(self.features):
            raise InputValidationError(
                f"Expected {len(self.features)} features, received {array.shape[1]}."
            )

        rows = [
            [
                feature.validate(row[index], coerce=self.coerce)
                for index, feature in enumerate(self.features)
            ]
            for row in array
        ]
        return InputBatch(np.asarray(rows), self.feature_names, is_single)


def process_untyped_array(data: Any) -> InputBatch:
    """Process array-only input when the user did not declare named features."""

    if _is_pandas_dataframe(data):
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise InputValidationError("Pandas DataFrame input cannot be empty.")
        names = tuple(str(column) for column in data.columns)
        return InputBatch(data, names, is_single=False)

    if isinstance(data, Mapping) or (
        isinstance(data, Sequence)
        and not isinstance(data, (str, bytes, np.ndarray))
        and len(data) > 0
        and all(isinstance(item, Mapping) for item in data)
    ):
        raise ConfigurationError(
            "Named input requires either features=[...] or an InputSchema."
        )
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
        is_single = True
    elif array.ndim == 2:
        is_single = False
    else:
        raise InputValidationError(
            f"Array input must be 1D or 2D, received shape {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise InputValidationError("Input array cannot be empty.")
    names = tuple(f"feature_{index}" for index in range(array.shape[1]))
    return InputBatch(array, names, is_single)


def _is_column(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)) or np.isscalar(value):
        return False
    try:
        len(value)
    except TypeError:
        return False
    return True


def _is_pandas_object(data: Any) -> bool:
    return type(data).__module__.split(".", 1)[0] == "pandas"


def _is_pandas_dataframe(data: Any) -> bool:
    return _is_pandas_object(data) and type(data).__name__ == "DataFrame"


def _pandas_to_supported(data: Any) -> Any:
    name = type(data).__name__
    if name == "DataFrame":
        return data.to_dict(orient="records")
    if name == "Series":
        return data.to_dict()
    raise InputValidationError(f"Unsupported pandas object: {name}.")
