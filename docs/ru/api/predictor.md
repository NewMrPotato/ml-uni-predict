# Predictor

{class}`~flexpredict.Predictor` — главная точка входа. Он хранит полный конвейер:
интерпретация входа → схема → preprocessing → engine → проверка выхода.

## Основные параметры

| Параметр | Когда использовать |
| --- | --- |
| `features` | выбрать и упорядочить именованные признаки |
| `schema` | проверить типы, defaults, validators и лишние поля |
| `preprocessor` | преобразовать значения перед моделью |
| `task` | явно задать regression/classification |
| `output_kind` | описать labels, probabilities или logits |
| `engine` | выбрать или передать движок |
| `engine_options` | задать device, dtype или настройки своего движка |
| `output_selector` | выбрать один элемент составного выхода |
| `name` | понятное имя в результате и ошибках ансамбля |

```python
predictor = Predictor(
    model,
    features=["age", "income"],
    task="regression",
    name="income-risk",
)
result = predictor.predict({"age": 31, "income": 150_000})
```

## Методы

- `predict(data)` — обычный прогноз;
- `predict_proba(data)` — вероятности классификации;
- `from_file(path, ...)` — загрузка полной модели и создание Predictor;
- `from_torch_weights(path, factory, ...)` — создание PyTorch-модели из state dict.

## Точная сигнатура

```{autoclass} flexpredict.Predictor
:members:
:undoc-members:
:noindex:
```
