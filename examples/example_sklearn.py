"""
Пример использования UniPredictor с sklearn моделью.
Демонстрирует: загрузку модели, создание предиктора, предсказание в разных форматах.
"""

import numpy as np
import joblib
from unipredict import UniPredictor

# 1. Загружаем модель (например, обученный RandomForestRegressor)
model = joblib.load('test_models/sklearn/rf_regressor.pkl')
feature_names = np.load('test_models/sklearn/feature_names_reg.npy', allow_pickle=True).tolist()
X = np.load('test_models/sklearn/X_reg.npy')

# 2. Создаём предиктор (без нормализации)
predictor = UniPredictor(
    model=model,
    feature_names=feature_names
)

print("=== SKLEARN ПРИМЕР ===")
print(f"Модель: {type(model).__name__}")
print(f"Признаки: {feature_names}")
print()

# 3. Предсказание для одного объекта (словарь)
sample = {name: X[0, i] for i, name in enumerate(feature_names)}
result1 = predictor.predict(sample)
print(f"1. Одиночный объект (словарь): {result1[0]:.4f}")

# 4. Предсказание для батча (словарь массивов)
batch_dict = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
result2 = predictor.predict(batch_dict)
print(f"2. Батч (словарь массивов): {result2}")

# 5. Предсказание из обычного массива (2D)
X_batch = X[:5]
result3 = predictor.predict(X_batch)
print(f"3. Батч (2D массив): {result3}")

# 6. Предсказание из списка словарей
list_of_dicts = [{name: X[i, j] for j, name in enumerate(feature_names)} for i in range(5)]
result4 = predictor.predict(list_of_dicts)
print(f"4. Батч (список словарей): {result4}")

print("\n✅ Пример sklearn завершён.")