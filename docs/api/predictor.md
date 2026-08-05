# Predictor

{class}`~flexpredict.Predictor` is the main entry point. It owns one complete pipeline:
input interpretation → schema → preprocessing → engine → output validation.

## Common constructor choices

| Option | Use it when |
| --- | --- |
| `features=[...]` | named fields need selection and ordering, but not strict type validation |
| `schema=InputSchema(...)` | input needs types, defaults, validators or strict extra-field handling |
| `preprocessor=...` | the model expects transformed values |
| `task=...` | task metadata cannot be inferred from the model |
| `output_kind=...` | the regular model output contains labels, probabilities or logits |
| `engine_options=...` | a framework needs settings such as PyTorch device and dtype |
| `output_selector=...` | the native model returns more than one output |

```python
predictor = Predictor(
    model,
    features=["age", "income"],
    task="regression",
    name="income-risk",
)

result = predictor.predict({"age": 31, "income": 150_000})
```

See [Core concepts](../concepts.md) for the lifecycle and
[Framework integrations](../guides/frameworks.md) for runtime-specific configuration.

## Complete API

```{autoclass} flexpredict.Predictor
:members:
:undoc-members:
:show-inheritance:
```
