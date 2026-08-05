# Входные данные и схемы

## Поддерживаемые формы

С `features` или схемой принимаются:

| Форма | Пример | Интерпретация |
| --- | --- | --- |
| 1D array-like | `[1, 2]` | один объект |
| 2D array-like | `[[1, 2], [3, 4]]` | пакет |
| словарь скаляров | `{"x1": 1, "x2": 2}` | один объект |
| словарь столбцов | `{"x1": [1, 3], "x2": [2, 4]}` | пакет |
| список записей | `[{"x1": 1, "x2": 2}, ...]` | пакет |
| structured NumPy array | именованные dtype-поля | пакет |
| pandas DataFrame | именованные столбцы | пакет |
| pandas Series | преобразуется в словарь | именованный вход |

Без `features` или схемы словарь запрещён: порядок его полей не должен неявно определять
порядок признаков модели.

## Выбор признаков

```python
predictor = Predictor(model, features=["income", "age"])

predictor.predict({
    "request_id": "71f2",
    "age": 31,
    "income": 150_000,
    "debug": True,
})
```

Модель получит сначала `income`, затем `age`; остальные поля будут отброшены. Все объявленные
признаки остаются обязательными.

## Строгая схема

```python
schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: 18 <= value <= 120,)),
    FeatureSpec("income", float, validators=(lambda value: value >= 0,)),
    FeatureSpec("score", float, default=0.5),
))

predictor = Predictor(model, schema=schema)
result = predictor.predict({"age": "31", "income": 150_000})
```

Возраст преобразуется в `int`, `score` получает default, а лишние поля запрещаются.
Валидаторы запускаются после преобразования типа.

`FeatureSpec` задаёт `dtype`, `required`, `nullable`, `default` и `validators`. Необязательный
признак без default превращается в `None`, поэтому ему обычно нужен `nullable=True`.

`coerce=False` проверяет безопасную совместимость NumPy dtype без вызова конструктора типа.
`extra_fields="ignore"` разрешает выбирать подмножество полей.

## Правила словаря столбцов

Все присутствующие объявленные поля должны быть либо скалярами, либо столбцами. Смешивание
запрещено. Столбцы должны иметь одинаковую ненулевую длину.

## pandas DataFrame

Generic/sklearn engine по возможности сохраняет DataFrame: порядок столбцов, index и dtype.
Это важно для `ColumnTransformer` и dtype selectors. `features=[...]` также сохраняет dtype,
поскольку не задаёт преобразование типов.

PyTorch и TensorFlow перед созданием тензоров преобразуют DataFrame в NumPy.

## Ошибки

- {class}`~flexpredict.MissingFeatureError` — нет обязательного поля;
- {class}`~flexpredict.UnexpectedFeatureError` — лишнее поле в строгой схеме;
- {class}`~flexpredict.InputValidationError` — неверное значение, форма или пустой вход;
- {class}`~flexpredict.ConfigurationError` — именованный вход использован без контракта.
