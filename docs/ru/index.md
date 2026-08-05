---
hide-toc: true
---

<div class="hero">

# Единый контракт инференса для любой модели

FlexPredict объединяет NumPy, scikit-learn, PyTorch, TensorFlow/Keras и пользовательские
модели: одинаковые входы, проверенные результаты и безопасные гетерогенные ансамбли.

```bash
python -m pip install flexpredict
```

</div>

FlexPredict — это библиотека композиции инференса. Она не обучает модели и не заменяет
model server. Библиотека связывает части, находящиеся непосредственно вокруг вызова модели:

- массивы, записи, словари столбцов и pandas DataFrame;
- выбор, порядок, проверку и предобработку признаков;
- выполнение в разных ML-фреймворках и загрузку артефактов;
- единый двумерный формат результата;
- ансамбли, участники которых могут использовать разные признаки и фреймворки.

```python
import numpy as np

from flexpredict import Predictor


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


predictor = Predictor(SumRegressor(), features=["x1", "x2"])
result = predictor.predict({"x1": [1, 3], "x2": [2, 4]})

print(result.values)  # [[3.0], [7.0]]
```

:::{note}
Версия 0.2 пока находится на стадии alpha. Она намеренно не сохраняет старый API
`unipredict`. Для production-окружения фиксируйте точную версию пакета.
:::

## С чего начать

**Первое знакомство:** [Установка и быстрый старт](getting-started.md), затем
[Основные понятия](concepts.md).

**Подключение данных приложения:** [Входные данные и схемы](guides/input-data.md) и
[Предобработка](guides/preprocessing.md).

**Интеграция модели:** [ML-фреймворки](guides/frameworks.md),
[Загрузка моделей](guides/model-loading.md) и [Свои движки](guides/engines.md).

**Объединение моделей:** [Ансамбли](guides/ensembles.md), включая разные наборы признаков,
веса, голосование и выравнивание классов.

```{toctree}
:hidden:
:maxdepth: 2

getting-started
concepts
guides/input-data
guides/preprocessing
guides/classification
guides/frameworks
guides/model-loading
guides/engines
guides/ensembles
guides/errors
security
api/index
changelog
contributing
```
