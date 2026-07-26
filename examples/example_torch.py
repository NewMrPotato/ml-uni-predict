"""
Пример использования UniPredictor с PyTorch моделью.
Демонстрирует: загрузку весов, нормализацию, выбор устройства.
"""

import numpy as np
import torch
from unipredict import UniPredictor

# Определяем класс модели (должен совпадать с тем, что использовался при обучении)
class SimpleRegressor(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 1. Загружаем модель и данные
model = SimpleRegressor()
model.load_state_dict(torch.load('test_models/torch/model_weights.pth'))
model.eval()

X = np.load('test_models/torch/X.npy')
feature_names = np.load('test_models/torch/feature_names.npy', allow_pickle=True).tolist()

# 2. Параметры нормализации (можно вычислить на тренировочных данных)
mean = X.mean(axis=0)
std = X.std(axis=0)

# 3. Создаём предиктор
predictor = UniPredictor(
    model=model,
    feature_names=feature_names,
    mean=mean,
    std=std,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("=== PYTORCH ПРИМЕР ===")
print(f"Модель: {type(model).__name__}")
print(f"Признаки: {feature_names}")
print(f"Device: {predictor.config.device}")
print(f"mean (первые 3): {mean[:3]}")
print(f"std (первые 3): {std[:3]}")
print()

# 4. Предсказание
sample = {name: X[0, i] for i, name in enumerate(feature_names)}
result = predictor.predict(sample)
print(f"Одиночный объект: {result[0]:.4f}")

batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
result_batch = predictor.predict(batch)
print(f"Батч (5 объектов): {result_batch.flatten()}")

print("\n✅ Пример PyTorch завершён.")