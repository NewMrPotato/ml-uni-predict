import pytest
import numpy as np
import joblib
import torch
import tensorflow as tf
import os


# Подавляем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SimpleRegressor(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

@pytest.fixture(scope="session")
def data_reg():
    """Регрессионные данные."""
    X = np.load('test_models/sklearn/X_reg.npy')
    y = np.load('test_models/sklearn/y_reg.npy')
    feature_names = np.load('test_models/sklearn/feature_names_reg.npy', allow_pickle=True).tolist()
    return X, y, feature_names

@pytest.fixture(scope="session")
def data_clf():
    """Классификационные данные."""
    X = np.load('test_models/sklearn/X_clf.npy')
    y = np.load('test_models/sklearn/y_clf.npy')
    feature_names = np.load('test_models/sklearn/feature_names_clf.npy', allow_pickle=True).tolist()
    return X, y, feature_names

@pytest.fixture(scope="session")
def sklearn_model():
    """Sklearn модель регрессии."""
    return joblib.load('test_models/sklearn/rf_regressor.pkl')

@pytest.fixture(scope="session")
def torch_model():
    """PyTorch модель регрессии."""
    model = SimpleRegressor()
    model.load_state_dict(torch.load('test_models/torch/model_weights.pth'))
    model.eval()
    return model

@pytest.fixture(scope="session")
def tf_model():
    """TensorFlow модель регрессии."""
    return tf.keras.models.load_model('test_models/tensorflow/model.keras')

@pytest.fixture(scope="session")
def norm_params(data_reg):
    """Параметры нормализации."""
    X, _, _ = data_reg
    return X.mean(axis=0), X.std(axis=0)

@pytest.fixture(scope="session")
def X_sample(data_reg):
    """Небольшая выборка данных для тестов движков."""
    X, _, _ = data_reg
    return X[:5]