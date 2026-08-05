# Custom engines and native outputs

## Selecting one model output

Some models return a mapping or tuple. Use `output_selector` to choose the prediction tensor:

```python
predictor = Predictor(model, output_selector="probabilities")
predictor = Predictor(model, output_selector=1)
```

A callable selector can implement more involved extraction:

```python
predictor = Predictor(
    model,
    output_selector=lambda output: output["head"]["scores"],
)
```

When a mapping has exactly one item, its value is selected automatically. A mapping with
multiple items requires an explicit selector. Selection errors become
{class}`~flexpredict.OutputValidationError`.

## Passing a custom engine instance

Subclass {class}`~flexpredict.InferenceEngine` when a model needs custom execution:

```python
import numpy as np

from flexpredict import InferenceEngine, Predictor


class RemoteEngine(InferenceEngine):
    def predict(self, values):
        response = self.model.invoke(np.asarray(values).tolist())
        return response["predictions"]


engine = RemoteEngine(client)
predictor = Predictor(client, engine=engine, task="regression")
```

Set `preserves_dataframe = True` if the engine intentionally accepts pandas DataFrames.
Otherwise FlexPredict converts the prepared input to NumPy. Do not pass `engine_options`
with an already-created engine instance.

## Registering an engine

Register a factory by name, optionally with an auto-detection function:

```python
from flexpredict import register_engine

register_engine(
    "remote",
    RemoteEngine,
    detector=lambda model: isinstance(model, RemoteClient),
    priority=100,
)

predictor = Predictor(remote_client)  # detected automatically
```

Detectors are evaluated from highest priority to lowest before built-in framework detection.
A detector failure is reported as a configuration error. Names cannot replace an existing
engine unless `replace=True` is explicitly supplied.

Factories receive the model, `output_selector` and all `engine_options`:

```python
predictor = Predictor(
    remote_client,
    engine="remote",
    engine_options={"timeout": 5.0},
)
```

## Error and output responsibilities

An engine may return native tensors or array-like values. FlexPredict converts objects using
the common `detach()`, `cpu()` and `numpy()` protocol before falling back to `np.asarray`.

Custom engines should raise library exceptions when they can provide a precise category.
Unexpected model errors from the built-in engines are wrapped in
{class}`~flexpredict.InferenceError`.

FlexPredict then validates dimensions, sample count, finite numeric values and result
semantics. A custom engine should not squeeze the batch dimension merely because a request
contains one sample.
