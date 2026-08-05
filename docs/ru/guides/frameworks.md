# Интеграции с ML-фреймворками

FlexPredict автоматически выбирает встроенный движок и не импортирует необязательные
фреймворки при загрузке базового пакета.

## NumPy и свои Python-модели

Generic engine принимает объект с `predict(X)`, сохраняет DataFrame и вызывает
`predict_proba(X)`, если такой метод есть. Задачу можно объявить через `_estimator_type` или
параметр `task`.

## scikit-learn

```bash
python -m pip install "flexpredict[sklearn,pandas]"
```

DataFrame передаётся с именами, порядком и dtype столбцов. FlexPredict понимает современные
sklearn tags, `_estimator_type`, `classes_` и `predict_proba()`. Полные `.joblib`, `.pkl` и
`.pickle` модели загружаются через `Predictor.from_file`.

## PyTorch

```bash
python -m pip install "flexpredict[torch]"
```

Для `torch.nn.Module` движок переносит модель на устройство, включает eval mode, создаёт
тензор нужного dtype, запускает `torch.inference_mode()` и возвращает NumPy.

```python
predictor = Predictor(
    torch_model,
    features=["x1", "x2"],
    task="regression",
    engine_options={"device": "cuda", "dtype": "float32"},
)
```

По умолчанию используются `cpu` и `float32`. У PyTorch нет общего probability API, поэтому
модель должна предоставить `predict_proba`, адаптер или probability-producing forward.

## TensorFlow и Keras

```bash
python -m pip install "flexpredict[tensorflow]"
```

`tf.keras.Model` определяется автоматически. Движок вызывает
`model.predict(np_values, verbose=0)`. Форматы `.keras`, `.h5` и `.hdf5` поддерживаются при
загрузке полной модели.

## Гетерогенные ансамбли

В одном ансамбле можно объединять PyTorch, sklearn, TensorFlow и свои модели. Совпадать
должны итоговые task, output kind, форма и классы, но не фреймворк или набор признаков.
