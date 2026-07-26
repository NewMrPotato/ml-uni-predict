import numpy as np
from typing import Any, Optional, Union, List
from .config import ModelConfig
from .processors import get_processor
from .engines import get_engine
from .exceptions import UniPredictError, DataFormatError


class UniPredictor:
    """
    Универсальный класс для инференса моделей машинного обучения.
    
    Поддерживает:
    - модели: sklearn, PyTorch, TensorFlow/Keras
    - входные данные: dict (один объект), dict массивов (батч), 
      список словарей, структурированный массив, обычный массив.
    - нормализацию, выбор устройства (CPU/CUDA)
    
    Пример:
        predictor = UniPredictor(model, feature_names=['x1', 'x2'])
        result = predictor.predict({'x1': 1.0, 'x2': 2.0})
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        """
        :param model: обученная модель (sklearn, torch.nn.Module, tf.keras.Model)
        :param config: объект ModelConfig (если не передан, создаётся из kwargs)
        :param kwargs: параметры для создания конфига (feature_names обязателен)
        """
        if config is None:
            config = ModelConfig(**kwargs)
        self.config = config
        self.feature_names = config.feature_names

        self._engine = get_engine(
            model, 
            device=config.device,
            engine_type=config.engine_type
        )

        self.mean = config.mean
        self.std = config.std

    def predict(self, data):
        """
        Основной метод предсказания.
        
        :param data: входные данные в одном из поддерживаемых форматов
        :return: np.ndarray предсказаний (n_samples, n_outputs) или (n_outputs,) для одного объекта
        """

        processor = get_processor(data, self.feature_names)
        X, is_batch = processor.process(data)

        if self.mean is not None and self.std is not None:
            if len(self.mean) != X.shape[1] or len(self.std) != X.shape[1]:
                raise DataFormatError("mean/std должны соответствовать числу признаков")
            X = (X - self.mean) / self.std

        result = self._engine.predict(X)

        if not is_batch and result.ndim == 2 and result.shape[0] == 1:
            result = result.flatten()

        return result