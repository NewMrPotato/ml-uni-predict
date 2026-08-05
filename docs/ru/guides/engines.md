# Свои движки и выходы моделей

## Выбор одного выхода

Если модель возвращает dict или tuple, укажите `output_selector`:

```python
Predictor(model, output_selector="probabilities")
Predictor(model, output_selector=1)
Predictor(model, output_selector=lambda output: output["head"]["scores"])
```

Значение единственного элемента dict выбирается автоматически. Для dict с несколькими
элементами selector обязателен.

## Экземпляр своего engine

```python
class RemoteEngine(InferenceEngine):
    def predict(self, values):
        response = self.model.invoke(np.asarray(values).tolist())
        return response["predictions"]


engine = RemoteEngine(client)
predictor = Predictor(client, engine=engine, task="regression")
```

Установите `preserves_dataframe = True`, если движок действительно понимает DataFrame. С
готовым экземпляром engine нельзя одновременно передавать `engine_options`.

## Регистрация

```python
register_engine(
    "remote",
    RemoteEngine,
    detector=lambda model: isinstance(model, RemoteClient),
    priority=100,
)
```

Пользовательские detectors выполняются по убыванию priority до встроенного определения
фреймворка. Существующее имя можно заменить только с `replace=True`.

Engine отвечает за нативный вызов. Predictor продолжает отвечать за вход, нормализацию формы
и проверку итогового результата.
