# Справочник API

Этот раздел нужен, когда сценарий уже понятен и требуются точные параметры, атрибуты или
исключения. Для последовательного объяснения начните с [руководств](../index.md#с-чего-начать).

Публичные объекты импортируются из корня пакета:

```python
from flexpredict import Predictor, InputSchema, EnsemblePredictor
```

## Какой объект выбрать

| Задача | Основной объект | Подробнее |
| --- | --- | --- |
| обернуть одну модель | {class}`~flexpredict.Predictor` | [Основные понятия](../concepts.md) |
| проверить именованные данные | {class}`~flexpredict.InputSchema`, {class}`~flexpredict.FeatureSpec` | [Входные данные](../guides/input-data.md) |
| обработать прогноз | {class}`~flexpredict.PredictionResult` | [Контракт результата](../getting-started.md#контракт-результата) |
| объединить модели | {class}`~flexpredict.EnsemblePredictor` | [Ансамбли](../guides/ensembles.md) |
| стандартизировать признаки | {class}`~flexpredict.Standardizer` | [Предобработка](../guides/preprocessing.md) |
| загрузить артефакт | `Predictor.from_file` или {func}`~flexpredict.load_model` | [Загрузка моделей](../guides/model-loading.md) |
| подключить runtime | {class}`~flexpredict.InferenceEngine`, {func}`~flexpredict.register_engine` | [Свои движки](../guides/engines.md) |
| обработать ошибку | {class}`~flexpredict.FlexPredictError` | [Ошибки](../guides/errors.md) |

```python
schema = InputSchema((
    FeatureSpec("age", int, validators=(lambda value: value >= 18,)),
    FeatureSpec("score", float),
))

predictor = Predictor(model, schema=schema, task="classification")
result = predictor.predict({"age": "31", "score": 0.82})
```

```{toctree}
:maxdepth: 2

predictor
schema
result
ensemble
engines
loading
preprocessing
exceptions
```
