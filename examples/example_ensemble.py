"""
Пример использования ансамблей: взвешенное среднее, медиана, пользовательская функция.
Демонстрирует создание EnsemblePredictor и его обёртывание в UniPredictor.
"""

import numpy as np
import joblib
import torch
import tensorflow as tf
from unipredict import UniPredictor, EnsemblePredictor

# Определяем класс для PyTorch модели
class SimpleRegressor(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

print("=== АНСАМБЛЬ ПРИМЕР ===\n")

# 1. Загружаем данные и создаём предикторы для трёх моделей
X = np.load('test_models/sklearn/X_reg.npy')
feature_names = np.load('test_models/sklearn/feature_names_reg.npy', allow_pickle=True).tolist()
mean = X.mean(axis=0)
std = X.std(axis=0)

# sklearn
sklearn_model = joblib.load('test_models/sklearn/rf_regressor.pkl')
p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)

# PyTorch
torch_model = SimpleRegressor()
torch_model.load_state_dict(torch.load('test_models/torch/model_weights.pth'))
torch_model.eval()
p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)

# TensorFlow
tf_model = tf.keras.models.load_model('test_models/tensorflow/model.keras')
p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)

# 2. Создаём ансамбль с взвешенным средним
ensemble_weighted = EnsemblePredictor(
    [p1, p2, p3],
    weights=[0.5, 0.3, 0.2],
    aggregation='weighted_mean'
)

# 3. Создаём ансамбль с медианой (устойчив к выбросам)
ensemble_median = EnsemblePredictor(
    [p1, p2, p3],
    aggregation='median'
)

# 4. Пользовательская функция: среднее геометрическое
def geometric_mean(preds, epsilon=1e-8):
    log_preds = np.log(np.abs(preds) + epsilon)
    return np.exp(np.mean(log_preds, axis=0))

ensemble_geo = EnsemblePredictor(
    [p1, p2, p3],
    aggregation=geometric_mean
)

# 5. Тестируем на одном объекте
sample = {name: X[0, i] for i, name in enumerate(feature_names)}

print("Предсказания отдельных моделей:")
print(f"  sklearn:  {p1.predict(sample)[0]:.4f}")
print(f"  PyTorch:  {p2.predict(sample)[0]:.4f}")
print(f"  TF:       {p3.predict(sample)[0]:.4f}")
print()

print("Ансамбли:")
print(f"  Взвешенное среднее: {ensemble_weighted.predict(sample)[0]:.4f}")
print(f"  Медиана:            {ensemble_median.predict(sample)[0]:.4f}")
print(f"  Среднее геометрическое: {ensemble_geo.predict(sample)[0]:.4f}")

# 6. Ансамбль можно обернуть в UniPredictor для единого интерфейса
final_predictor = UniPredictor(
    model=ensemble_weighted,
    feature_names=feature_names
)
print(f"\nАнсамбль, обёрнутый в UniPredictor: {final_predictor.predict(sample)[0]:.4f}")

print("\n✅ Пример ансамбля завершён.")