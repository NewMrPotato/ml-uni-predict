# Model loading

Use the `Predictor` class methods for the shortest load-and-wrap path. Use the functions below
when artifact loading and predictor construction need to happen separately.

| Artifact | Recommended API |
| --- | --- |
| joblib/pickle complete model | `Predictor.from_file(path)` |
| Keras complete model | `Predictor.from_file(path)` |
| PyTorch state dict | `Predictor.from_torch_weights(path, factory)` |
| trusted complete PyTorch model | `Predictor.from_file(path, loader="torch_model")` |

See [Loading model artifacts](../guides/model-loading.md) for safe defaults and serialization
risks.

## `load_model`

```{autofunction} flexpredict.load_model
```

## `load_torch_state_dict`

```{autofunction} flexpredict.load_torch_state_dict
```
