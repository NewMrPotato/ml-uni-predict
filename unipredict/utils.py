import torch
import joblib
import tensorflow as tf
from pathlib import Path
from .exceptions import ModelLoadingError


def load_model_from_path(model_path: str, engine_type: str = 'auto'):
    """
    Загружает модель из файла или папки, определяя тип по расширению.
    
    :param model_path: путь к модели
    :param engine_type: 'sklearn', 'torch', 'tensorflow' или 'auto'
    :return: загруженная модель
    """

    path = Path(model_path)

    if engine_type == 'torch':
        return torch.load(path)
    elif engine_type == 'sklearn':
        return joblib.load(path)
    elif engine_type == 'tensorflow':
        return tf.keras.models.load_model(path)
    else:
        
        if path.suffix == '.pth' or path.suffix == '.pt':
            return torch.load(path)
        elif path.suffix in ['.pkl', '.joblib']:
            return joblib.load(path)
        elif path.is_dir() and (path / 'saved_model.pb').exists():
            return tf.keras.models.load_model(path)
        else:
            raise ModelLoadingError(f"Не удалось определить тип модели: {model_path}")