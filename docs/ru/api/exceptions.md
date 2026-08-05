# Иерархия исключений

Узкое исключение удобно перехватывать, когда приложение умеет исправить конкретную ситуацию.
На общей границе можно перехватить {class}`~flexpredict.FlexPredictError`.

```python
try:
    result = predictor.predict(payload)
except InputValidationError as exc:
    return {"error": "invalid_input", "detail": str(exc)}
except FlexPredictError as exc:
    return {"error": "prediction_failed", "detail": str(exc)}
```

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

## Классы

```{automodule} flexpredict.exceptions
:members:
:show-inheritance:
:noindex:
```
