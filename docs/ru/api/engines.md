# Inference engines

Большинству пользователей этот API не нужен. Он предназначен для runtime, который нельзя
подключить через generic, PyTorch или TensorFlow engine.

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

Engine отвечает за нативный вызов модели. Predictor отвечает за вход, нормализацию формы и
проверку результата.

## `InferenceEngine`

```{autoclass} flexpredict.InferenceEngine
:members:
:undoc-members:
:show-inheritance:
:noindex:
```

## `register_engine`

```{autofunction} flexpredict.register_engine
:noindex:
```
