"""
Пример использования UniPredictor с TensorFlow моделью.
Демонстрирует: загрузку модели из .keras, нормализацию.
"""

import numpy as np
import tensorflow as tf
from unipredict import UniPredictor

# 1. Загружаем модель
model = tf.keras.models.load_model('test_models/tensorflow/model.keras')

X = np.load('test_models/tensorflow/X.npy')
feature_names = np.load('test_models/tensorflow/feature_names.npy', allow_pickle=True).tolist()

# 2. Нормализация
mean = X.mean(axis=0)
std = X.std(axis=0)

# 3. Создаём предиктор
predictor = UniPredictor(
    model=model,
    feature_names=feature_names,
    mean=mean,
    std=std
)

print("=== TENSORFLOW ПРИМЕР ===")
print(f"Модель: {type(model).__name__}")
print(f"Признаки: {feature_names}")
print()

# 4. Предсказание
sample = {name: X[0, i] for i, name in enumerate(feature_names)}
result = predictor.predict(sample)
print(f"Одиночный объект: {result[0]:.4f}")

batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
result_batch = predictor.predict(batch)
print(f"Батч (5 объектов): {result_batch.flatten()}")

print("\n✅ Пример TensorFlow завершён.")