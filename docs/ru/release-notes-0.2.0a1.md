# FlexPredict 0.2.0a1

Первый alpha-релиз переработанного слоя композиции инференса FlexPredict. Он намеренно не
сохраняет старый API `unipredict`.

## Главное

- Единый `Predictor` для NumPy, sklearn-совместимых, PyTorch, TensorFlow/Keras и своих
  моделей с ленивым импортом фреймворков.
- Выбор именованных признаков и явная проверка `InputSchema` для словарей, записей, массивов,
  structured arrays и pandas.
- Сохранение имён и dtype DataFrame для sklearn `Pipeline` и `ColumnTransformer`.
- Callable, transformer-based и встроенная стандартизация.
- Двумерный `PredictionResult` с явными task, output kind и классами.
- Mean, weighted mean, median, min, max и voting, включая выравнивание порядка классов и
  нормализованные глобальные веса из `.npy`.
- Загрузка joblib, Keras, доверенных полных PyTorch-моделей и безопасных PyTorch state dict.
- Строгая проверка формы, конечности, dtype выхода и совместимости ансамбля.

## Установка

```bash
python -m pip install "flexpredict==0.2.0a1"
```

Устанавливайте только нужные интеграции:

```bash
python -m pip install "flexpredict[sklearn,pandas]==0.2.0a1"
```

До версии 1.0 публичный API ещё может меняться. Загружайте модельные артефакты только из
доверенных источников.

- [Пакет PyPI](https://pypi.org/project/flexpredict/0.2.0a1/)
- [Релиз GitHub](https://github.com/NewMrPotato/ml-flex-predict/releases/tag/v0.2.0a1)
