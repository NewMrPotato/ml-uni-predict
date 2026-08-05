# Preprocessing

Each predictor can own a different preprocessor. FlexPredict accepts either a callable or an
object with `transform(X)`, including sklearn-compatible transformers.

```python
from flexpredict import Predictor

predictor = Predictor(
    model,
    features=["age", "income"],
    preprocessor=fitted_transformer,
)
```

Preprocessing runs after schema selection and validation and before framework conversion.
The preprocessor must:

- return a two-dimensional batch;
- preserve the number of samples;
- return at least one feature;
- avoid NaN and infinite numeric values.

It may change the number of feature columns. Errors are wrapped in
{class}`~flexpredict.PreprocessingError` while retaining the original exception as the cause.

## Built-in standardization

{class}`~flexpredict.Standardizer` applies column-wise `(X - mean) / std`:

```python
from flexpredict import Predictor, Standardizer

predictor = Predictor(
    model,
    features=["x1", "x2", "x3"],
    preprocessor=Standardizer(
        mean=[10.0, 20.0, 30.0],
        std=[2.0, 5.0, 10.0],
    ),
)
```

Its parameters must be one-dimensional, finite, equal in length and non-empty. Every
standard deviation must be greater than zero. At inference time the input width must match
the parameter length.

## Callable preprocessing

```python
import numpy as np


def add_ratio(values):
    values = np.asarray(values, dtype=float)
    ratio = values[:, [0]] / np.maximum(values[:, [1]], 1e-12)
    return np.column_stack([values, ratio])


predictor = Predictor(model, features=["a", "b"], preprocessor=add_ratio)
```

If DataFrame semantics are needed inside preprocessing, the callable can operate directly on
the selected DataFrame. Its result is normalized to a NumPy array before inference.

## Per-model pipelines

An ensemble sends the original request to every member. Each member independently selects
features and applies its own preprocessing:

```python
predictor_a = Predictor(
    model_a,
    features=["age", "income"],
    preprocessor=scaler_a,
)
predictor_b = Predictor(
    model_b,
    features=["score", "history_length"],
    preprocessor=scaler_b,
)
```

This is preferable to applying one global transformation when models were trained with
different feature contracts.
