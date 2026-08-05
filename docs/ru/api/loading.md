# Загрузка артефактов

Методы Predictor удобнее для «загрузить и сразу обернуть». Функции нужны, когда этапы должны
быть разделены.

| Артефакт | Рекомендуемый API |
| --- | --- |
| joblib/pickle | `Predictor.from_file(path)` |
| полная Keras-модель | `Predictor.from_file(path)` |
| PyTorch state dict | `Predictor.from_torch_weights(path, factory)` |
| доверенная полная PyTorch-модель | `Predictor.from_file(path, loader="torch_model")` |

## `load_model`

Загружает полную модель через выбранный или определённый по расширению loader.

```{autofunction} flexpredict.load_model
:noindex:
```

## `load_torch_state_dict`

Создаёт модель через factory и применяет безопасно загруженный state dict.

```{autofunction} flexpredict.load_torch_state_dict
:noindex:
```
