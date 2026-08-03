# FlexPredict

[![PyPI](https://img.shields.io/pypi/v/flexpredict)](https://pypi.org/project/flexpredict/)
[![Python](https://img.shields.io/pypi/pyversions/flexpredict)](https://pypi.org/project/flexpredict/)
[![CI](https://github.com/NewMrPotato/ml-flex-predict/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NewMrPotato/ml-flex-predict/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/flexpredict)](https://github.com/NewMrPotato/ml-flex-predict/blob/main/LICENSE)

FlexPredict is a schema-aware inference composition layer for Python with unified outputs,
preprocessing, model loading and heterogeneous ensemble support. It gives NumPy,
scikit-learn-compatible, PyTorch, TensorFlow/Keras and custom models a consistent input and
output contract while allowing every model to keep its own schema and preprocessing pipeline.

> The current public release is [0.2.0a1](https://pypi.org/project/flexpredict/0.2.0a1/).
> FlexPredict 0.2 is an alpha redesign and intentionally does not preserve the old
> `unipredict` API.

## Why FlexPredict?

Application inputs rarely look exactly like a model matrix. A service may receive one
JSON object, a dictionary of columns or a list of records, while different models in an
ensemble require different normalization, devices and output handling. FlexPredict
composes those concerns without becoming a training framework or a model server.

Core properties:

- zero-configuration inference for array inputs;
- named and schema-validated tabular inputs;
- callable, sklearn-style and built-in preprocessing;
- canonical prediction results with shape `(n_samples, n_outputs)`;
- lazy optional dependencies for PyTorch and TensorFlow;
- safe regression and classification ensembles;
- global ensemble weights from lists, arrays or `.npy` files;
- a registry for custom inference engines.

## Installation

FlexPredict supports Python 3.10 through 3.13. Install the current alpha release from PyPI:

```bash
python -m pip install flexpredict
```

Pin the version for a reproducible application environment:

```bash
python -m pip install "flexpredict==0.2.0a1"
```

The base package installs only NumPy. Add the integrations required by the application:

```bash
python -m pip install "flexpredict[sklearn,pandas]"
python -m pip install "flexpredict[torch]"
python -m pip install "flexpredict[tensorflow]"
```

| Extra | Installs | Use it for |
| --- | --- | --- |
| `pandas` | pandas | DataFrame inputs and dtype-preserving pipelines |
| `sklearn` | scikit-learn and joblib | sklearn estimators and joblib artifacts |
| `torch` | PyTorch | tensor inference and PyTorch model loading |
| `tensorflow` | TensorFlow | Keras/TensorFlow inference and model loading |
| `all` | every optional integration | environments that need every supported framework |

Optional frameworks are imported lazily: installing the base package does not require or
import PyTorch, TensorFlow, pandas or scikit-learn. Contributors should use an editable
source install described in the
[contribution guide](https://github.com/NewMrPotato/ml-flex-predict/blob/main/CONTRIBUTING.md),
not the commands above.

## Quick start

### Array input with no configuration

```python
import numpy as np

from flexpredict import Predictor

class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


predictor = Predictor(SumRegressor())
result = predictor.predict([[1.0, 2.0], [3.0, 4.0]])

print(result.values)  # [[3.0], [7.0]] — always a 2D NumPy array
print(result.task)    # regression
```

### Output contract

`PredictionResult.values` is always a non-empty two-dimensional array. FlexPredict rejects
non-finite numeric output and applies the following dtype rules:

| Output kind | Supported values |
| --- | --- |
| Regression or generic `values` | finite real integers and floats |
| `probabilities` | finite real numbers between 0 and 1 |
| `logits` | finite real numbers |
| Classification `labels` | strings, booleans and finite real numbers |

Complex numbers and arbitrary Python objects such as dictionaries, nested lists and `None`
are rejected. A custom classifier returning string labels should declare its semantics when
they cannot be inferred:

```python
predictor = Predictor(
    custom_classifier,
    task="classification",
    output_kind="labels",
)
```

### Named input

```python
predictor = Predictor(model, features=["age", "income", "score"])

result = predictor.predict({
    "age": 31,
    "income": 150_000,
    "score": 0.82,
})

print(result.single())
```

The same predictor accepts a dictionary of columns or a list of records:

```python
predictor.predict({
    "age": [31, 45],
    "income": [150_000, 90_000],
    "score": [0.82, 0.61],
})

predictor.predict([
    {"age": 31, "income": 150_000, "score": 0.82},
    {"age": 45, "income": 90_000, "score": 0.61},
])
```

`features=[...]` is a feature-selection shortcut: declared features are required, selected
from a named input and passed to the model in the declared order; unrelated fields are
ignored. This allows one shared request to feed ensemble members with different feature
subsets. Use an explicit `InputSchema` when extra fields must be rejected.

Pandas `DataFrame` input remains a DataFrame for generic and sklearn-compatible models.
Column names, their declared order and pandas dtypes are preserved, so sklearn pipelines
using `ColumnTransformer` with named columns or dtype selectors continue to work. PyTorch
and TensorFlow engines convert DataFrame input to a NumPy matrix before tensor inference.

## Schema-aware input

Use an explicit schema only when validation is useful:

```python
from flexpredict import FeatureSpec, InputSchema, Predictor

schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: 18 <= value <= 120,)),
    FeatureSpec("income", float, validators=(lambda value: value >= 0,)),
    FeatureSpec("score", float, default=0.5),
))

predictor = Predictor(model, schema=schema)
result = predictor.predict({"age": "31", "income": 150_000})
```

By default, values are coerced to their declared types and extra fields are rejected for an
explicit schema.
Errors identify the invalid or missing feature before the model is called.

## Preprocessing

FlexPredict accepts a callable or an object with `transform(X)`:

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

Each member of an ensemble can use a different schema and preprocessor.

## Ensembles and weights files

`aggregation_weights` are coefficients controlling the contribution of each model.
They are not the trained parameters inside a neural network.

```python
from flexpredict import EnsemblePredictor

ensemble = EnsemblePredictor(
    predictors=[predictor_a, predictor_b, predictor_c],
    aggregation="weighted_mean",
    aggregation_weights="models/global_weights.npy",
)

result = ensemble.predict(data)
```

Weights files are loaded with `np.load(path, allow_pickle=False)`. The array must be
one-dimensional, contain one finite non-negative weight per predictor and have a positive
sum. Weights are normalized automatically and exposed as the read-only
`ensemble.aggregation_weights` array in predictor order. A Python list or NumPy array can
be passed instead of a file path.

Every member receives the original input independently, so its schema can select and order
features for that model. Members must still agree on task, output kind, batch shape and
classification classes before aggregation. Errors identify the failing member by index and
name. Majority-vote ties are deterministic: the prediction from the earliest member wins.

## Loading model artifacts

Complete sklearn/joblib and Keras artifacts can be turned into predictors directly:

```python
sklearn_predictor = Predictor.from_file(
    "models/random_forest.joblib",
    features=["age", "income", "score"],
)

keras_predictor = Predictor.from_file(
    "models/network.keras",
    features=["age", "income", "score"],
)
```

PyTorch state dicts need a model factory because weights alone do not describe the
Python architecture:

```python
torch_predictor = Predictor.from_torch_weights(
    "models/network_weights.pth",
    model_factory=lambda: MyNetwork(input_size=3),
    features=["age", "income", "score"],
    task="regression",
    engine_options={"device": "cuda", "dtype": "float32"},
)
```

PyTorch weights are loaded with `weights_only=True`. A complete trusted PyTorch model can
be loaded explicitly with `loader="torch_model"`; `.pt` and `.pth` are never guessed
because the same extensions are commonly used for incompatible artifact types.

Supported regression aggregations are `mean`, `weighted_mean`, `median`, `min`, `max`
and a callable. Classification probabilities can be averaged after verifying class
metadata. Class labels require the explicit `voting` strategy.

## Classification

```python
predictor = Predictor(
    classifier,
    features=["age", "income", "score"],
    task="classification",  # inferred for sklearn classifiers
)

labels = predictor.predict(data)
probabilities = predictor.predict_proba(data)

print(labels.values)
print(probabilities.values)
print(probabilities.classes)
```

For a PyTorch or Keras model whose regular forward output already contains
probabilities, configure `output_kind="probabilities"`.

## Engine options

Framework-specific runtime settings stay grouped and optional:

```python
predictor = Predictor(
    torch_model,
    features=["x1", "x2", "x3"],
    task="regression",
    engine_options={
        "device": "cuda",
        "dtype": "float32",
    },
)
```

Unknown engine options are rejected instead of being silently ignored.

Custom `InferenceEngine` implementations receive NumPy input by default. An engine that
intentionally supports pandas semantics can declare `preserves_dataframe = True`.

## Development

Bug reports, focused feature proposals, documentation improvements and pull requests are
welcome. See the
[contribution guide](https://github.com/NewMrPotato/ml-flex-predict/blob/main/CONTRIBUTING.md)
for editable installation, branch policy, required checks and pull-request expectations.
Maintainers use the
[release checklist](https://github.com/NewMrPotato/ml-flex-predict/blob/main/docs/release-checklist.md)
for TestPyPI and PyPI publication.

## Security

Only load model artifacts and preprocessing objects from trusted sources. Python pickle,
joblib and some framework serialization formats may execute code while loading. NumPy
ensemble-weight files are always loaded with pickle support disabled.

## License

MIT
