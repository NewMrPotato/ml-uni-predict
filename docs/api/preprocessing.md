# Preprocessing

`Standardizer` is a small dependency-free transformer implementing `(X - mean) / std`.

```python
standardizer = Standardizer(mean=[10, 20], std=[2, 5])
predictor = Predictor(model, preprocessor=standardizer)
```

For sklearn transformers or custom callables, pass them directly to `Predictor`; they do not
need to inherit from this class. See [Preprocessing](../guides/preprocessing.md).

## Complete API

```{autoclass} flexpredict.Standardizer
:members:
:undoc-members:
```
