# Предобработка

`Standardizer` — небольшой трансформер без дополнительных зависимостей, выполняющий
`(X - mean) / std`.

```python
standardizer = Standardizer(mean=[10, 20], std=[2, 5])
predictor = Predictor(model, preprocessor=standardizer)
```

sklearn transformers и callables передаются напрямую в Predictor и не должны наследоваться
от Standardizer.

## Точная сигнатура

```{autoclass} flexpredict.Standardizer
:members:
:undoc-members:
:noindex:
```
