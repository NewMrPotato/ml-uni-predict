# Классификация

## sklearn-классификаторы

Для совместимых sklearn-моделей FlexPredict автоматически читает тип задачи и `classes_`:

```python
predictor = Predictor(classifier, features=["age", "score"])

labels = predictor.predict(data)
probabilities = predictor.predict_proba(data)

print(labels.output_kind)         # labels
print(probabilities.output_kind)  # probabilities
print(probabilities.classes)      # класс каждого столбца
```

## Вероятности из обычного forward

Нейросети часто возвращают вероятности обычным вызовом и не имеют `predict_proba`:

```python
predictor = Predictor(
    probability_model,
    features=["x1", "x2"],
    task="classification",
    output_kind="probabilities",
)
result = predictor.predict_proba(data)
```

Тогда `predict()` и `predict_proba()` используют обычный путь модели. Значения проверяются на
конечность и диапазон `[0, 1]`.

Для логитов задайте `output_kind="logits"`. FlexPredict намеренно не применяет softmax:
бинарные и многоклассовые задачи требуют разных преобразований.

## Метки и классы

Метки могут быть строками, bool, целыми или конечными вещественными числами. Словари,
вложенные контейнеры, `None`, bytes, complex и нечисловые объекты запрещены.

`classes` — непустой одномерный массив уникальных меток. Для probabilities и logits его
длина должна совпадать с числом выходных столбцов.

## Ансамбли классификаторов

Вероятности и логиты агрегируются численно. Если одинаковые классы перечислены в разном
порядке, FlexPredict переставит столбцы по порядку первого участника.

Для готовых меток используется `aggregation="voting"`. При равенстве голосов побеждает
метка самого раннего участника.
