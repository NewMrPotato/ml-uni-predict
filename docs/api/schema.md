# Input schema

Use {class}`~flexpredict.FeatureSpec` to describe one field and
{class}`~flexpredict.InputSchema` to define their model input order.

```python
schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: 18 <= value <= 120,)),
    FeatureSpec("score", float, default=0.5),
))
```

| `FeatureSpec` field | Meaning |
| --- | --- |
| `name` | external field name and position in model input |
| `dtype` | conversion target, or `None` to preserve the value |
| `required` | whether an absent field is an error |
| `nullable` | whether an explicit `None` is accepted |
| `default` | value inserted when the field is absent |
| `validators` | callables run after conversion |

`MISSING` distinguishes “no default” from a real default such as `None`.
See [Input data and schemas](../guides/input-data.md) for supported containers and edge cases.

## `FeatureSpec`

```{autoclass} flexpredict.FeatureSpec
:members:
:undoc-members:
```

## `InputSchema`

```{autoclass} flexpredict.InputSchema
:members:
:undoc-members:
```

## `MISSING`

```{autodata} flexpredict.MISSING
```
