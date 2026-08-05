# Схема входных данных

`FeatureSpec` описывает поле, а `InputSchema` задаёт порядок полей для модели.

```python
schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: 18 <= value <= 120,)),
    FeatureSpec("score", float, default=0.5),
))
```

| Поле `FeatureSpec` | Смысл |
| --- | --- |
| `name` | внешнее имя и позиция во входе модели |
| `dtype` | функция преобразования или `None` |
| `required` | обязательно ли присутствие поля |
| `nullable` | допустим ли явный `None` |
| `default` | значение при отсутствии поля |
| `validators` | проверки после преобразования |

`MISSING` отличает отсутствие default от настоящего default, равного `None`.

## `FeatureSpec`

```{autoclass} flexpredict.FeatureSpec
:members:
:undoc-members:
:noindex:
```

## `InputSchema`

```{autoclass} flexpredict.InputSchema
:members:
:undoc-members:
:noindex:
```

## `MISSING`

```{autodata} flexpredict.MISSING
:noindex:
```
