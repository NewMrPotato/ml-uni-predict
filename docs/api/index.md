# API reference

Use this section when you already know the workflow and need exact signatures, attributes or
exceptions. For task-oriented explanations, start with the [guides](../index.md#choose-a-path).

All supported public objects are exported from `flexpredict`:

```python
from flexpredict import Predictor, InputSchema, EnsemblePredictor
```

Avoid importing implementation helpers from submodules unless you are extending the engine
system.

## Find the right object

| I want to… | Start with | Related guide |
| --- | --- | --- |
| wrap one model | {class}`~flexpredict.Predictor` | [Core concepts](../concepts.md) |
| validate named input | {class}`~flexpredict.InputSchema` and {class}`~flexpredict.FeatureSpec` | [Input data](../guides/input-data.md) |
| consume a prediction | {class}`~flexpredict.PredictionResult` | [Result contract](../getting-started.md#the-result-contract) |
| combine models | {class}`~flexpredict.EnsemblePredictor` | [Ensembles](../guides/ensembles.md) |
| standardize features | {class}`~flexpredict.Standardizer` | [Preprocessing](../guides/preprocessing.md) |
| load an artifact | {meth}`~flexpredict.Predictor.from_file` or {func}`~flexpredict.load_model` | [Model loading](../guides/model-loading.md) |
| support another runtime | {class}`~flexpredict.InferenceEngine` and {func}`~flexpredict.register_engine` | [Custom engines](../guides/engines.md) |
| handle a failure | {class}`~flexpredict.FlexPredictError` | [Errors](../guides/errors.md) |

## Typical composition

```python
from flexpredict import FeatureSpec, InputSchema, Predictor

schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: value >= 18,)),
    FeatureSpec("score", float),
))

predictor = Predictor(model, schema=schema, task="classification")
result = predictor.predict({"age": "31", "score": 0.82})
```

```{toctree}
:maxdepth: 2

predictor
schema
result
ensemble
engines
loading
preprocessing
exceptions
```
