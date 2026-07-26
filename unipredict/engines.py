from abc import ABC, abstractmethod
import numpy as np
import torch
import tensorflow as tf
from .exceptions import ModelLoadingError


class InferenceEngine(ABC):
    """Базовый класс для движка инференса."""
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Принимает 2D numpy-массив (n_samples, n_features), возвращает predictions."""
        pass

class SklearnEngine(InferenceEngine):
    def __init__(self, model):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class TorchEngine(InferenceEngine):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(tensor)
        return output.cpu().numpy()

class TensorFlowEngine(InferenceEngine):
    def __init__(self, model):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)



def get_engine(model, device='cpu', engine_type=None):
    
    if engine_type == 'sklearn':
        return SklearnEngine(model)
    elif engine_type == 'torch':
        return TorchEngine(model, device)
    elif engine_type == 'tensorflow':
        return TensorFlowEngine(model)
    elif engine_type is not None:
        raise ValueError(f"Неизвестный тип движка: {engine_type}. Доступны: 'sklearn', 'torch', 'tensorflow'.")
    
    # Автоопределение
    if isinstance(model, torch.nn.Module):
        return TorchEngine(model, device)
    elif isinstance(model, tf.keras.Model):
        return TensorFlowEngine(model)
    elif hasattr(model, 'predict') and callable(model.predict):
        return SklearnEngine(model)
    else:
        raise ValueError(
            "Не удалось определить тип модели. Модель не является torch.nn.Module, "
            "tf.keras.Model и не имеет метода predict(). "
            "Пожалуйста, укажите engine_type явно ('sklearn', 'torch', 'tensorflow')."
        )