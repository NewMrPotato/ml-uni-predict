# Inference engines

Most users do not need this extension API. Use it when a model cannot be represented by the
built-in generic, PyTorch or TensorFlow engines.

```python
class MyEngine(InferenceEngine):
    def predict(self, values):
        return self.model.run(values)


register_engine(
    "my-runtime",
    MyEngine,
    detector=lambda model: isinstance(model, MyRuntimeModel),
    priority=100,
)
```

An engine is responsible for native execution. `Predictor` remains responsible for input
preparation, shape normalization and final output validation. See
[Custom engines and outputs](../guides/engines.md) before implementing an adapter.

## `InferenceEngine`

```{autoclass} flexpredict.InferenceEngine
:members:
:undoc-members:
:show-inheritance:
```

## `register_engine`

```{autofunction} flexpredict.register_engine
```
