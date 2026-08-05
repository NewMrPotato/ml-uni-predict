# Установка и быстрый старт

## Требования

FlexPredict поддерживает Python 3.10–3.13. Базовой установке нужен только NumPy:

```bash
python -m pip install flexpredict
```

Для воспроизводимости зафиксируйте текущую alpha-версию:

```bash
python -m pip install "flexpredict==0.2.0a1"
```

Дополнительные зависимости устанавливаются только для нужных интеграций:

| Extra | Что устанавливает | Для чего нужен |
| --- | --- | --- |
| `pandas` | pandas | DataFrame и сохранение dtype |
| `sklearn` | scikit-learn, joblib | sklearn-модели и joblib-артефакты |
| `torch` | PyTorch | инференс и загрузка PyTorch |
| `tensorflow` | TensorFlow | Keras/TensorFlow |
| `all` | всё перечисленное | окружения со всеми интеграциями |

```bash
python -m pip install "flexpredict[sklearn,pandas]"
python -m pip install "flexpredict[torch]"
python -m pip install "flexpredict[tensorflow]"
```

Фреймворки импортируются лениво: обычный `import flexpredict` не требует PyTorch,
TensorFlow, pandas или scikit-learn.

## Инференс массива без конфигурации

Generic engine принимает любой объект с методом `predict(X)`:

```python
import numpy as np

from flexpredict import Predictor


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


predictor = Predictor(SumRegressor())
single = predictor.predict([1, 2, 3])
batch = predictor.predict([[1, 2], [3, 4]])

print(single.single())  # 6.0
print(batch.values)     # [[3.0], [7.0]]
```

Одномерный вход считается одним объектом, двумерный — пакетом, даже если в нём одна строка.

## Именованные данные

`features` разрешает словари и записи, выбирает нужные поля и передаёт их модели в заданном
порядке. Посторонние поля игнорируются.

```python
predictor = Predictor(SumRegressor(), features=["x1", "x2"])

predictor.predict({"request_id": "abc", "x2": 2, "x1": 1})
predictor.predict({"x1": [1, 3], "x2": [2, 4]})
predictor.predict([
    {"x1": 1, "x2": 2},
    {"x1": 3, "x2": 4},
])
```

Если нужны типы, значения по умолчанию, валидаторы и запрет лишних полей, используйте
{class}`~flexpredict.InputSchema`.

## Контракт результата

{class}`~flexpredict.PredictionResult` не зависит от фреймворка:

- `values` всегда имеет форму `(n_samples, n_outputs)`;
- `task` — `regression`, `classification` или `unknown`;
- `output_kind` различает значения, метки, вероятности и логиты;
- `classes` связывает столбцы с классами;
- `is_single` хранит информацию об одиночном входе;
- `model_name` называет модель или ансамбль.

Для одного объекта можно вызвать `result.single()`. В пакетном коде лучше использовать
стабильный двумерный `result.values`.
