import numpy as np
import pytest

from flexpredict import (
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

