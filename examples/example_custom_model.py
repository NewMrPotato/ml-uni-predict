"""
Пример использования UniPredictor с кастомной моделью (не sklearn, не torch, не tf).
Демонстрирует: создание класса с методом predict и явное указание engine_type.
"""

import numpy as np
from unipredict import UniPredictor

# 1. Определяем кастомный класс модели
class CustomModel:
    def __init__(self, coef=2.0):
        self.coef = coef
    
    def predict(self, X):
        # Простая линейная функция: y = coef * sum(X)
        return self.coef * X.sum(axis=1)

# 2. Создаём модель и данные
model = CustomModel(coef=3.5)
feature_names = ['x1', 'x2', 'x3']
X_sample = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 3. Создаём предиктор.
# Поскольку у модели есть метод predict, автоопределение сработает как SklearnEngine.
predictor = UniPredictor(
    model=model,
    feature_names=feature_names
)

print("=== КАСТОМНАЯ МОДЕЛЬ ПРИМЕР ===")
print(f"Модель: {type(model).__name__}")
print(f"coef = {model.coef}")
print()

# 4. Предсказание
sample = {name: X_sample[0, i] for i, name in enumerate(feature_names)}
result = predictor.predict(sample)
print(f"Одиночный объект: {result[0]:.4f} (ожидается {model.coef * 6.0:.4f})")

batch = {name: X_sample[:, i].tolist() for i, name in enumerate(feature_names)}
result_batch = predictor.predict(batch)
print(f"Батч: {result_batch} (ожидается {model.coef * np.array([6.0, 15.0])})")

print("\n✅ Пример кастомной модели завершён.")