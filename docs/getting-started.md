# Installation and quick start

## Requirements

FlexPredict supports Python 3.10 through 3.13. The base installation depends only on NumPy:

```bash
python -m pip install flexpredict
```

The current release is an alpha. Pin it when reproducibility matters:

```bash
python -m pip install "flexpredict==0.2.0a1"
```

Install only the integrations your application uses:

| Extra | Packages | Purpose |
| --- | --- | --- |
| `pandas` | pandas | DataFrame inputs and dtype-preserving pipelines |
| `sklearn` | scikit-learn, joblib | sklearn-compatible estimators and joblib artifacts |
| `torch` | PyTorch | tensor inference and PyTorch loading |
| `tensorflow` | TensorFlow | Keras/TensorFlow inference and model loading |
| `all` | all of the above | environments that need every integration |

```bash
python -m pip install "flexpredict[sklearn,pandas]"
python -m pip install "flexpredict[torch]"
python -m pip install "flexpredict[tensorflow]"
```

Optional frameworks are imported lazily. Importing `flexpredict` does not import or require
PyTorch, TensorFlow, pandas or scikit-learn.

## Zero-configuration array inference

Any object with a callable `predict(X)` method can use the generic engine:

```python
import numpy as np

from flexpredict import Predictor


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


predictor = Predictor(SumRegressor())

single = predictor.predict([1, 2, 3])
batch = predictor.predict([[1, 2], [3, 4]])

print(single.single())  # 6.0
print(batch.values)     # [[3.0], [7.0]]
```

A one-dimensional input is one sample. A two-dimensional input is always a batch, even if
it has one row.

## Named input

Declare `features` to accept mappings and records. FlexPredict selects the declared fields,
orders them for the model and ignores unrelated fields:

```python
predictor = Predictor(SumRegressor(), features=["x1", "x2"])

predictor.predict({"request_id": "abc", "x2": 2, "x1": 1})
predictor.predict({"x1": [1, 3], "x2": [2, 4]})
predictor.predict([
    {"x1": 1, "x2": 2},
    {"x1": 3, "x2": 4},
])
```

Use an explicit {class}`~flexpredict.InputSchema` when you need types, defaults, validators,
nullable fields or strict rejection of extra fields.

## The result contract

{class}`~flexpredict.PredictionResult` is independent of the model framework:

- `values` always has shape `(n_samples, n_outputs)`;
- `task` is `regression`, `classification` or `unknown`;
- `output_kind` describes values, labels, probabilities or logits;
- `classes` carries class-column metadata when available;
- `is_single` remembers whether the input represented one record;
- `model_name` identifies the producing predictor or ensemble.

Use `result.single()` for a scalar or one-dimensional copy when the result contains one
sample. Keep `result.values` for predictable batch-oriented application code.

## Next steps

- [Understand the composition pipeline](concepts.md).
- [Choose the right input contract](guides/input-data.md).
- [Connect a supported ML framework](guides/frameworks.md).
- [Build a heterogeneous ensemble](guides/ensembles.md).
