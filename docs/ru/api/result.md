# Результат прогноза

Каждая модель возвращает один и тот же объект независимо от фреймворка.

```python
result = predictor.predict([1.0, 2.0])

print(result.values.shape)  # (1, n_outputs)
print(result.n_samples)     # 1
value = result.single()
```

| Атрибут | Контракт |
| --- | --- |
| `values` | непустой массив `(n_samples, n_outputs)` |
| `task` | `regression`, `classification` или `unknown` |
| `output_kind` | `values`, `labels`, `probabilities` или `logits` |
| `classes` | класс каждого probability/logit столбца |
| `is_single` | был ли вход одним объектом |
| `model_name` | имя модели или ансамбля |

`single()` возвращает скаляр для одного столбца или копию 1D строки для нескольких.

## Точная сигнатура

```{autoclass} flexpredict.PredictionResult
:members:
:undoc-members:
:noindex:
```
