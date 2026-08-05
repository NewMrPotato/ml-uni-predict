# Errors and troubleshooting

All library-specific exceptions inherit from {class}`~flexpredict.FlexPredictError`.

| Exception | Meaning |
| --- | --- |
| `ConfigurationError` | invalid predictor, schema, engine, loader or ensemble setup |
| `InputValidationError` | data violates the input contract |
| `MissingFeatureError` | required named field is absent |
| `UnexpectedFeatureError` | strict schema received an extra field |
| `PreprocessingError` | preprocessing failed or returned an invalid batch |
| `EngineNotAvailableError` | optional framework is not installed |
| `InferenceError` | model execution failed |
| `EnsembleInferenceError` | identified ensemble member failed |
| `OutputValidationError` | native output, result or aggregation violates its contract |
| `UnsupportedOutputError` | requested output method is unavailable |
| `EnsembleCompatibilityError` | member results cannot be combined safely |

## Named input requires a contract

If a dictionary is passed to a predictor created without `features` or `schema`, add one of
them. FlexPredict will not guess feature order from mapping iteration order.

## A model has multiple outputs

Configure `output_selector` when a model returns a dict with multiple keys or a tuple whose
prediction item is not the complete value. See [Custom engines and outputs](engines.md).

## `predict_proba()` is unavailable

Generic and PyTorch engines require the model to expose `predict_proba()`, unless the
predictor is configured with `output_kind="probabilities"`. For a neural model, either
describe probability-producing forward output explicitly or provide an adapter that applies
the correct sigmoid/softmax semantics.

## An ensemble rejects class metadata

Every probability or logit member must expose the same unique class set. sklearn classifiers
normally provide `classes_`. For custom predictors, construct `PredictionResult` with a
matching `classes` array.

## DataFrame pipelines lose named semantics

The generic/sklearn engine preserves DataFrames. PyTorch and TensorFlow engines intentionally
convert them to arrays. A FlexPredict preprocessor also normalizes its returned value to
NumPy. Keep an sklearn `ColumnTransformer` inside the sklearn model pipeline or ensure that a
custom engine declares `preserves_dataframe = True`.

## Optional dependency is missing

Install the targeted extra shown in the exception:

```bash
python -m pip install "flexpredict[sklearn]"
python -m pip install "flexpredict[torch]"
python -m pip install "flexpredict[tensorflow]"
```
