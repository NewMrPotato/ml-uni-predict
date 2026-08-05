# Ensemble predictor

The ensemble calls each member with the original input, checks result compatibility and then
aggregates an array shaped `(n_predictors, n_samples, n_outputs)`.

```python
ensemble = EnsemblePredictor(
    [predictor_a, predictor_b],
    aggregation="weighted_mean",
    aggregation_weights=[0.7, 0.3],
    name="production-ensemble",
)
result = ensemble.predict(data)
```

| Aggregation | Suitable output |
| --- | --- |
| `mean`, `weighted_mean`, `median`, `min`, `max` | numeric regression, probability or logit output |
| `voting` | one classification label per sample |
| callable | custom shape-preserving logic |

See [Ensembles](../guides/ensembles.md) for weights, class alignment and heterogeneous member
examples.

## Complete API

```{autoclass} flexpredict.EnsemblePredictor
:members:
:undoc-members:
```
