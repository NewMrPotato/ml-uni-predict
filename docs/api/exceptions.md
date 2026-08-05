# Exception hierarchy

Catch a narrow exception when the caller can recover from a specific condition. Catch
{class}`~flexpredict.FlexPredictError` at an application boundary when all library failures
share one response or logging policy.

```python
from flexpredict import FlexPredictError, InputValidationError

try:
    result = predictor.predict(payload)
except InputValidationError as exc:
    return {"error": "invalid_input", "detail": str(exc)}
except FlexPredictError as exc:
    return {"error": "prediction_failed", "detail": str(exc)}
```

See [Errors and troubleshooting](../guides/errors.md) for likely causes and fixes.

```text
FlexPredictError
├── ConfigurationError
├── InputValidationError
│   ├── MissingFeatureError
│   └── UnexpectedFeatureError
├── PreprocessingError
├── EngineNotAvailableError
├── InferenceError
│   └── EnsembleInferenceError
├── OutputValidationError
├── UnsupportedOutputError
└── EnsembleCompatibilityError
```

```{automodule} flexpredict.exceptions
:members:
:show-inheritance:
```
