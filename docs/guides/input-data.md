# Input data and schemas

## Supported forms

With `features` or an explicit schema, FlexPredict accepts:

| Form | Example | Single or batch |
| --- | --- | --- |
| 1D array-like | `[1, 2]` | single sample |
| 2D array-like | `[[1, 2], [3, 4]]` | batch |
| scalar mapping | `{"x1": 1, "x2": 2}` | single sample |
| mapping of columns | `{"x1": [1, 3], "x2": [2, 4]}` | batch |
| list of records | `[{"x1": 1, "x2": 2}, ...]` | batch |
| structured NumPy array | named dtype fields | batch; one row is marked single |
| pandas DataFrame | named columns | batch |
| pandas Series | converted to a mapping | single or column-oriented by values |

Without named features or a schema, only array-like and DataFrame inputs are accepted.
Passing a mapping without a named contract raises {class}`~flexpredict.ConfigurationError`
because column order would otherwise be ambiguous.

## Feature selection

The concise form selects and orders a subset of a shared request:

```python
from flexpredict import Predictor

predictor = Predictor(model, features=["income", "age"])

result = predictor.predict({
    "request_id": "71f2",
    "age": 31,
    "income": 150_000,
    "debug": True,
})
```

The model receives `income` first and `age` second. `request_id` and `debug` are ignored.
Every declared feature is still required.

## A strict schema

```python
from flexpredict import FeatureSpec, InputSchema, Predictor

schema = InputSchema((
    FeatureSpec(
        "age",
        int,
        validators=(lambda value: 18 <= value <= 120,),
    ),
    FeatureSpec(
        "income",
        float,
        validators=(lambda value: value >= 0,),
    ),
    FeatureSpec("score", float, default=0.5),
))

predictor = Predictor(model, schema=schema)
result = predictor.predict({"age": "31", "income": 150_000})
```

The string age is coerced to `int`, the default score is inserted and extra fields are
rejected. Validators run after coercion. A validator may return `False` to reject a value;
returning `None` is treated as success, which is useful for validators that raise their own
errors.

### Feature options

`FeatureSpec` provides:

- `dtype`: callable conversion target, or `None` to preserve values;
- `required`: whether a missing value is an error;
- `nullable`: whether an explicit `None` is allowed;
- `default`: value used when the field is absent;
- `validators`: callables evaluated in order after conversion.

An optional feature without a default resolves to `None`; it therefore normally also needs
`nullable=True`.

Set `coerce=False` on the schema to validate safe NumPy dtype compatibility without calling
the declared type. Set `extra_fields="ignore"` to select a subset instead of rejecting extra
keys.

## Mapping-of-columns rules

All present declared fields in a mapping must be either scalars or columns. Mixing a column
with a scalar is rejected, and columns must have equal non-zero lengths. Defaults are applied
row by row.

```python
# Valid batch
schema.process({"age": [20, 30], "income": [50_000, 80_000]})

# Invalid: a column mixed with a scalar
schema.process({"age": [20, 30], "income": 50_000})
```

## DataFrame preservation

The generic/sklearn engine preserves a pandas DataFrame when possible, including selected
column order, index and dtypes. This is important for sklearn pipelines that use named
`ColumnTransformer` columns or dtype selectors.

`features=[...]` uses `dtype=None`, so selected DataFrames retain their dtypes. A schema with
coercing dtypes validates record-by-record and reconstructs a DataFrame. PyTorch and
TensorFlow engines convert DataFrames to NumPy before creating tensors.

## Common validation failures

- {class}`~flexpredict.MissingFeatureError`: a required feature is absent;
- {class}`~flexpredict.UnexpectedFeatureError`: a strict schema sees an extra field;
- {class}`~flexpredict.InputValidationError`: invalid value, shape, nullability, mixed column
  layout, duplicate DataFrame columns or empty input;
- {class}`~flexpredict.ConfigurationError`: named input was used without a named contract.
