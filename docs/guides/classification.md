# Classification

## sklearn-compatible classifiers

For sklearn and compatible estimators, FlexPredict reads task metadata and `classes_`
automatically:

```python
from flexpredict import Predictor

predictor = Predictor(classifier, features=["age", "score"])

labels = predictor.predict({"age": [20, 40], "score": [0.2, 0.8]})
probabilities = predictor.predict_proba(
    {"age": [20, 40], "score": [0.2, 0.8]}
)

print(labels.output_kind)         # labels
print(probabilities.output_kind)  # probabilities
print(probabilities.classes)      # class for each output column
```

`predict()` uses the model's label-producing method. `predict_proba()` uses the engine's
probability method.

## Framework-native probability output

Many neural networks return probabilities from their regular forward pass and do not define
`predict_proba`. Describe that contract explicitly:

```python
predictor = Predictor(
    probability_model,
    features=["x1", "x2"],
    task="classification",
    output_kind="probabilities",
)

result = predictor.predict_proba(data)
```

When `output_kind="probabilities"`, both `predict()` and `predict_proba()` use the engine's
regular prediction path. Probability values are checked to be finite and between zero and
one.

For a model whose forward pass emits logits, use `output_kind="logits"` with `predict()`.
FlexPredict does not apply softmax automatically, because binary and multiclass models may
require different conversions. Provide an adapter, a custom engine or a model-level
`predict_proba()` implementation when probabilities are needed.

## Labels and class metadata

Classification labels may be strings, booleans, integers or finite real numbers. Arbitrary
objects, nested containers, `None`, bytes, complex values and non-finite floats are rejected.

`classes` must be a non-empty one-dimensional array of unique supported labels. For
probability and logit results, the number of classes must equal the output width.

## Classification ensembles

Use numeric aggregation for probabilities or logits:

```python
ensemble = EnsemblePredictor(
    [predictor_a, predictor_b],
    aggregation="weighted_mean",
    aggregation_weights=[0.6, 0.4],
)
result = ensemble.predict_proba(data)
```

If members expose the same classes in different orders, FlexPredict reorders probability or
logit columns to match the first predictor. It will not reorder label outputs because a label
column does not encode per-class positions.

For label predictions, use `aggregation="voting"`. Voting supports one label per sample.
Ties are resolved deterministically in predictor order: the earliest tied prediction wins.
