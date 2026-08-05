# Предобработка

Predictor принимает callable или объект с `transform(X)`, включая sklearn-трансформеры.
Предобработка выполняется после схемы и до вызова движка.

```python
predictor = Predictor(
    model,
    features=["age", "income"],
    preprocessor=fitted_transformer,
)
```

Результат предобработки должен быть двумерным, сохранять число объектов, содержать хотя бы
один признак и не иметь NaN/Infinity. Число столбцов меняться может. Ошибки оборачиваются в
{class}`~flexpredict.PreprocessingError`.

## `Standardizer`

{class}`~flexpredict.Standardizer` выполняет `(X - mean) / std` по столбцам:

```python
predictor = Predictor(
    model,
    features=["x1", "x2", "x3"],
    preprocessor=Standardizer(
        mean=[10.0, 20.0, 30.0],
        std=[2.0, 5.0, 10.0],
    ),
)
```

`mean` и `std` должны быть конечными одномерными массивами одинаковой ненулевой длины.
Все значения `std` должны быть положительными.

## Callable

```python
def add_ratio(values):
    values = np.asarray(values, dtype=float)
    ratio = values[:, [0]] / np.maximum(values[:, [1]], 1e-12)
    return np.column_stack([values, ratio])
```

Каждый участник ансамбля может иметь собственные признаки и preprocessing. Это правильнее
одной глобальной трансформации, если модели обучались на разных представлениях данных.
