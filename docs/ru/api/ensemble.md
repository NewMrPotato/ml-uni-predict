# EnsemblePredictor

EnsemblePredictor передаёт исходный input каждому участнику, проверяет совместимость и
агрегирует массив `(n_predictors, n_samples, n_outputs)`.

```python
ensemble = EnsemblePredictor(
    [predictor_a, predictor_b],
    aggregation="weighted_mean",
    aggregation_weights=[0.7, 0.3],
    name="production-ensemble",
)
```

| Агрегация | Подходящий выход |
| --- | --- |
| `mean`, `weighted_mean`, `median`, `min`, `max` | числа, probabilities или logits |
| `voting` | одна метка класса на объект |
| callable | собственная логика с сохранением формы |

## Точная сигнатура

```{autoclass} flexpredict.EnsemblePredictor
:members:
:undoc-members:
:noindex:
```
