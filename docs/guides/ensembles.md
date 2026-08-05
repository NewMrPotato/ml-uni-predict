# Ensembles

An {class}`~flexpredict.EnsemblePredictor` combines predictor-like objects that return
{class}`~flexpredict.PredictionResult`.

## Regression aggregation

```python
from flexpredict import EnsemblePredictor

ensemble = EnsemblePredictor(
    [predictor_a, predictor_b, predictor_c],
    aggregation="weighted_mean",
    aggregation_weights=[0.5, 0.3, 0.2],
    name="risk-ensemble",
)

result = ensemble.predict(data)
```

Supported built-in strategies are:

| Strategy | Requirement | Behavior |
| --- | --- | --- |
| `mean` | numeric outputs | arithmetic mean; default |
| `weighted_mean` | numeric outputs | global normalized member weights |
| `median` | numeric outputs | element-wise median |
| `min` | numeric outputs | element-wise minimum |
| `max` | numeric outputs | element-wise maximum |
| `voting` | one label per sample | deterministic majority vote |

## Global weights

Weights are coefficients for entire predictors, not trained neural-network parameters. Pass a
sequence, NumPy array or `.npy` path:

```python
ensemble = EnsemblePredictor(
    predictors,
    aggregation="weighted_mean",
    aggregation_weights="models/global_weights.npy",
)
```

Weights must be one-dimensional, finite, non-negative, match the number of predictors and
have a positive sum. FlexPredict normalizes them and exposes the resulting read-only array as
both `aggregation_weights` and `weights`.

When `weighted_mean` has no explicit weights, all members receive equal weights. Files must
use `.npy` and are loaded with `allow_pickle=False`.

## Different features and frameworks

Every member receives the original input, then independently selects features and preprocesses
them:

```python
ensemble = EnsemblePredictor([
    Predictor(sklearn_model, features=["age", "income"]),
    Predictor(torch_model, features=["score", "history"], task="regression"),
])

result = ensemble.predict({
    "age": 31,
    "income": 150_000,
    "score": 0.82,
    "history": 14,
})
```

This also allows members to use the same features in different orders.

## Compatibility checks

Before aggregation, all members must agree on:

- task;
- output kind;
- complete value shape;
- whether the request was interpreted as a single sample;
- class metadata.

For probability and logit outputs, members may list the same classes in different orders.
FlexPredict aligns their columns to the first member. A missing class, extra class or missing
class metadata is rejected.

## Classification voting

```python
ensemble = EnsemblePredictor(
    [classifier_a, classifier_b, classifier_c],
    aggregation="voting",
)
labels = ensemble.predict(data)
```

Voting requires `output_kind="labels"` and exactly one output column. If multiple labels tie,
the label returned by the earliest tied member wins.

To average probabilities, call `predict_proba()` and use a numeric strategy instead.

## Custom aggregation

A callable receives an array with shape
`(n_predictors, n_samples, n_outputs)` and must return exactly
`(n_samples, n_outputs)`:

```python
import numpy as np


def trimmed_mean(stacked):
    ordered = np.sort(stacked, axis=0)
    return ordered[1:-1].mean(axis=0)


ensemble = EnsemblePredictor(predictors, aggregation=trimmed_mean)
```

Custom aggregation cannot be combined with `aggregation_weights`. Returned numeric values
must be finite.

## Diagnosing member failures

If a member raises during inference, FlexPredict raises
{class}`~flexpredict.EnsembleInferenceError` with its zero-based index and configured name.
Set meaningful predictor names to make operational errors actionable.
