# Загрузка моделей

## sklearn/joblib

```python
predictor = Predictor.from_file(
    "models/random_forest.joblib",
    features=["age", "income", "score"],
)
```

Расширения `.joblib`, `.pkl` и `.pickle` выбирают joblib. Дополнительные аргументы загрузчика
передаются через `loader_options`.

## Keras

```python
predictor = Predictor.from_file(
    "models/network.keras",
    features=["age", "income", "score"],
    task="classification",
    output_kind="probabilities",
    loader_options={"compile": False},
)
```

## PyTorch state dict

State dict не описывает Python-архитектуру, поэтому нужен factory:

```python
predictor = Predictor.from_torch_weights(
    "models/network_weights.pth",
    model_factory=lambda: MyNetwork(input_size=3),
    features=["age", "income", "score"],
    task="regression",
    engine_options={"device": "cuda", "dtype": "float32"},
)
```

По умолчанию используется `weights_only=True`, а `map_location` берётся из устройства
predictor. Ключ `state_dict` определяется автоматически; для другого вложенного ключа есть
`state_dict_key`. `strict` передаётся в `load_state_dict`.

## Полная PyTorch-модель

`.pt` и `.pth` намеренно не определяются автоматически: там может быть state dict, полная
модель или checkpoint приложения.

```python
predictor = Predictor.from_file(
    "models/trusted_model.pt",
    loader="torch_model",
    task="regression",
)
```

Этот путь использует `weights_only=False` и способен выполнить код при десериализации.
Загружайте только доверенные файлы; state dict предпочтительнее.

Неизвестные расширения не угадываются. Для них нужно явно передать `loader`.
