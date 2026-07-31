# FlexPredict

FlexPredict is a small, composable inference layer for tabular machine-learning models.
It gives NumPy, scikit-learn-compatible, PyTorch, TensorFlow/Keras and custom models a
consistent input and output contract, while allowing every model to keep its own schema
and preprocessing pipeline.

> FlexPredict 0.2 is an alpha redesign. It intentionally does not preserve the old
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

Development install:

```bash
git clone https://github.com/NewMrPotato/ml-flex-predict.git
cd ml-flex-predict
pip install -e .
```

Optional frameworks:

```bash
pip install -e ".[sklearn]"
pip install -e ".[torch]"
pip install -e ".[tensorflow]"
pip install -e ".[pandas]"
pip install -e ".[dev]"
```

The base package depends only on NumPy. Importing `flexpredict` does not import or
require PyTorch, TensorFlow, pandas or scikit-learn.

## Quick start

### Array input with no configuration

```python
from flexpredict import Predictor

predictor = Predictor(model)
result = predictor.predict([[1.0, 2.0], [3.0, 4.0]])

print(result.values)       # always a 2D NumPy array
print(result.task)         # inferred for sklearn-compatible estimators
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

By default, values are coerced to their declared types and extra fields are rejected.
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
sum. Weights are normalized automatically. A Python list or NumPy array can be passed
instead of a file path.

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

## Development

```bash
python -m pytest
python -m pytest --cov=flexpredict --cov-fail-under=85
python -m ruff check flexpredict tests examples
python -m mypy flexpredict
python -m compileall -q flexpredict tests examples
python -m build
```

The fast unit suite requires only NumPy and pytest. Framework integration suites are
kept separate so that schema and ensemble tests do not require heavy optional packages.
The current fast suite enforces at least 85% coverage; this threshold will increase as
the optional-framework integration matrix grows.

## Security

Only load model artifacts and preprocessing objects from trusted sources. Python pickle,
joblib and some framework serialization formats may execute code while loading. NumPy
ensemble-weight files are always loaded with pickle support disabled.

## License

MIT
