# Prediction result

Every prediction returns the same framework-independent object. Prefer the stable 2D
`values` array in batch code; use `single()` only when one sample is expected.

```python
result = predictor.predict([1.0, 2.0])

print(result.values.shape)  # (1, n_outputs)
print(result.n_samples)     # 1
value = result.single()     # scalar or 1D copy
```

| Attribute | Contract |
| --- | --- |
| `values` | non-empty `(n_samples, n_outputs)` array |
| `task` | `regression`, `classification` or `unknown` |
| `output_kind` | `values`, `labels`, `probabilities` or `logits` |
| `classes` | optional class label for every probability/logit column |
| `is_single` | whether the input represented one record |
| `model_name` | producing predictor or ensemble name |

The constructor validates shapes, dtypes, finite values, probability range, class uniqueness
and task/output compatibility.

## Complete API

```{autoclass} flexpredict.PredictionResult
:members:
:undoc-members:
```
