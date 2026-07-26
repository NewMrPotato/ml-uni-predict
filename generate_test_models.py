"""Скрипт для генерации тестовых моделей."""
import numpy as np
import joblib
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs('test_models/sklearn', exist_ok=True)
os.makedirs('test_models/torch', exist_ok=True)
os.makedirs('test_models/tensorflow', exist_ok=True)


print("=== Генерация тестовых данных ===")
np.random.seed(42)

X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
feature_names_reg = ['x1', 'x2', 'x3', 'x4', 'x5']

X_clf, y_clf = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=42)
feature_names_clf = ['f1', 'f2', 'f3', 'f4', 'f5']

print("=== Создание sklearn моделей ===")
rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)
rf_reg.fit(X_reg, y_reg)
joblib.dump(rf_reg, 'test_models/sklearn/rf_regressor.pkl')
print("✅ sklearn RandomForestRegressor сохранен")

rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X_clf, y_clf)
joblib.dump(rf_clf, 'test_models/sklearn/rf_classifier.pkl')
print("✅ sklearn RandomForestClassifier сохранен")

np.save('test_models/sklearn/X_reg.npy', X_reg)
np.save('test_models/sklearn/y_reg.npy', y_reg)
np.save('test_models/sklearn/X_clf.npy', X_clf)
np.save('test_models/sklearn/y_clf.npy', y_clf)
np.save('test_models/sklearn/feature_names_reg.npy', feature_names_reg)
np.save('test_models/sklearn/feature_names_clf.npy', feature_names_clf)

print("=== Создание PyTorch модели ===")

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

torch_model = SimpleRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)

X_torch = torch.tensor(X_reg, dtype=torch.float32)
y_torch = torch.tensor(y_reg, dtype=torch.float32).reshape(-1, 1)

for epoch in range(100):
    optimizer.zero_grad()
    pred = torch_model(X_torch)
    loss = criterion(pred, y_torch)
    loss.backward()
    optimizer.step()

torch.save(torch_model, 'test_models/torch/model_full.pt')
torch.save(torch_model.state_dict(), 'test_models/torch/model_weights.pth')
print(f"✅ PyTorch модель сохранена (final loss: {loss.item():.4f})")

np.save('test_models/torch/X.npy', X_reg)
np.save('test_models/torch/y.npy', y_reg)
np.save('test_models/torch/feature_names.npy', feature_names_reg)

print("=== Создание TensorFlow модели ===")

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer='adam', loss='mse')
tf_model.fit(X_reg, y_reg, epochs=50, verbose=0)

tf_model.save('test_models/tensorflow/model.keras')
tf_model.save('test_models/tensorflow/model.h5')
print("✅ TensorFlow модель сохранена (форматы .keras и .h5)")

np.save('test_models/tensorflow/X.npy', X_reg)
np.save('test_models/tensorflow/y.npy', y_reg)
np.save('test_models/tensorflow/feature_names.npy', feature_names_reg)

print("\n=== ВСЕ МОДЕЛИ СОЗДАНЫ ===")