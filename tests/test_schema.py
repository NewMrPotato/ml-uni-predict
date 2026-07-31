import numpy as np
import pytest

from flexpredict import (
    ConfigurationError,
    FeatureSpec,
    InputSchema,
    InputValidationError,
    MissingFeatureError,
    UnexpectedFeatureError,
)


def test_schema_processes_single_mapping_in_declared_order():
    schema = InputSchema.from_names(["x1", "x2"])

    batch = schema.process({"x2": 2, "x1": 1})

    assert batch.is_single
    assert batch.feature_names == ("x1", "x2")
    assert np.allclose(batch.values, [[1.0, 2.0]])


def test_schema_processes_mapping_of_columns():
    schema = InputSchema.from_names(["x1", "x2"])

    batch = schema.process({"x1": [1, 2], "x2": np.array([3, 4])})

    assert not batch.is_single
    assert np.allclose(batch.values, [[1, 3], [2, 4]])


def test_schema_processes_list_of_mappings_and_structured_array():
    schema = InputSchema.from_names(["x1", "x2"])
    records = [{"x1": 1, "x2": 2}, {"x1": 3, "x2": 4}]
    structured = np.array([(1, 2), (3, 4)], dtype=[("x1", "i4"), ("x2", "i4")])

    assert np.allclose(schema.process(records).values, [[1, 2], [3, 4]])
    assert np.allclose(schema.process(structured).values, [[1, 2], [3, 4]])


def test_schema_applies_defaults_coercion_and_validators():
    schema = InputSchema(
        (
            FeatureSpec("age", int, validators=(lambda value: value >= 18,)),
            FeatureSpec("score", float, default=0.5),
        )
    )

    batch = schema.process({"age": "21"})

    assert np.allclose(batch.values.astype(float), [[21, 0.5]])


def test_schema_rejects_missing_extra_mixed_and_empty_inputs():
    schema = InputSchema.from_names(["x1", "x2"])

    with pytest.raises(MissingFeatureError):
        schema.process({"x1": 1})
    with pytest.raises(UnexpectedFeatureError):
        schema.process({"x1": 1, "x2": 2, "unknown": 3})
    with pytest.raises(InputValidationError, match="cannot mix"):
        schema.process({"x1": [1, 2], "x2": 3})
    with pytest.raises(InputValidationError, match="empty"):
        schema.process({})
    with pytest.raises(InputValidationError, match="empty"):
        schema.process([])


def test_schema_validates_array_width():
    schema = InputSchema.from_names(["x1", "x2"])

    with pytest.raises(InputValidationError, match="Expected 2 features"):
        schema.process(np.array([1, 2, 3]))


def test_schema_can_ignore_extra_fields():
    schema = InputSchema.from_names(["x1"], extra_fields="ignore")

    batch = schema.process({"extra": 10, "x1": 2})

    assert batch.values[0, 0] == 2


def test_feature_nullable_and_validator_failures():
    nullable = InputSchema((FeatureSpec("value", float, nullable=True),))
    invalid = InputSchema((FeatureSpec("value", int, validators=(lambda value: False,)),))

    assert nullable.process({"value": None}).values[0, 0] is None
    with pytest.raises(InputValidationError, match="failed validation"):
        invalid.process({"value": 1})


def test_feature_type_coercion_failure_is_contextual():
    schema = InputSchema.from_names(["value"], dtype=float)

    with pytest.raises(InputValidationError, match="cannot be converted"):
        schema.process({"value": "not-a-number"})


def test_schema_configuration_is_validated():
    with pytest.raises(ConfigurationError, match="at least one"):
        InputSchema(())
    with pytest.raises(ConfigurationError, match="unique"):
        InputSchema((FeatureSpec("x"), FeatureSpec("x")))
    with pytest.raises(ConfigurationError, match="extra_fields"):
        InputSchema((FeatureSpec("x"),), extra_fields="invalid")
    with pytest.raises(ConfigurationError, match="non-empty"):
        FeatureSpec("")


def test_schema_rejects_column_length_and_array_dimension_errors():
    schema = InputSchema.from_names(["x1", "x2"])

    with pytest.raises(InputValidationError, match="same length"):
        schema.process({"x1": [1], "x2": [2, 3]})
    with pytest.raises(InputValidationError, match="1D or 2D"):
        schema.process(np.ones((1, 1, 2)))
    with pytest.raises(InputValidationError, match="empty"):
        schema.process(np.empty((0, 2)))
