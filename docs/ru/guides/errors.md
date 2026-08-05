# Ошибки и диагностика

Все исключения библиотеки наследуются от {class}`~flexpredict.FlexPredictError`.

| Исключение | Что означает |
| --- | --- |
| `ConfigurationError` | неверная конфигурация predictor, schema, engine, loader или ensemble |
| `InputValidationError` | вход нарушает контракт |
| `MissingFeatureError` | нет обязательного поля |
| `UnexpectedFeatureError` | лишнее поле строгой схемы |
| `PreprocessingError` | ошибка или неверный результат preprocessing |
| `EngineNotAvailableError` | не установлен дополнительный фреймворк |
| `InferenceError` | модель завершилась с ошибкой |
| `EnsembleInferenceError` | завершился с ошибкой конкретный участник |
| `OutputValidationError` | неверная форма или семантика результата |
| `UnsupportedOutputError` | модель не поддерживает запрошенный выход |
| `EnsembleCompatibilityError` | результаты нельзя безопасно объединить |

## Частые ситуации

**Named input requires a contract:** добавьте `features` или `schema`; порядок словаря не
используется как неявный порядок модели.

**Multiple named outputs:** задайте `output_selector`.

**Нет `predict_proba`:** объявите forward как `output_kind="probabilities"` только если он
действительно возвращает вероятности, либо добавьте корректный адаптер.

**Несовместимые классы ансамбля:** у каждого probability/logit результата должен быть один и
тот же уникальный набор `classes`.

**Не установлен фреймворк:** установите соответствующий extra — `sklearn`, `torch` или
`tensorflow`.
