import numpy as np
from typing import List, Dict, Any, Union, Optional
from .exceptions import DataFormatError, FeatureNotFoundError


class BaseProcessor:
    """Базовый класс для процессоров."""
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def process(self, data):
        """Возвращает (X: np.ndarray, is_batch: bool)"""
        raise NotImplementedError



class DictSingleProcessor(BaseProcessor):
    def process(self, data: Dict[str, Any]):
        try:
            values = [data[name] for name in self.feature_names]
        except KeyError as e:
            raise FeatureNotFoundError(f"Отсутствует признак: {e}")
        X = np.array([values], dtype=np.float32)
        return X, False

class DictBatchProcessor(BaseProcessor):
    def process(self, data: Dict[str, Union[List, np.ndarray]]):
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) != 1:
            raise DataFormatError("Все массивы в словаре должны иметь одинаковую длину")
        n_samples = lengths[0]
        X = np.zeros((n_samples, len(self.feature_names)), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if name not in data:
                raise FeatureNotFoundError(f"Отсутствует признак: {name}")
            arr = np.asarray(data[name])
            if len(arr) != n_samples:
                raise DataFormatError(f"Длина массива для '{name}' не совпадает")
            X[:, i] = arr
        return X, True

class ListOfDictsProcessor(BaseProcessor):
    def process(self, data: List[Dict[str, Any]]):
        X = []
        for item in data:
            try:
                row = [item[name] for name in self.feature_names]
            except KeyError as e:
                raise FeatureNotFoundError(f"Отсутствует признак: {e}")
            X.append(row)
        return np.array(X, dtype=np.float32), True

class StructuredArrayProcessor(BaseProcessor):
    def process(self, data: np.ndarray):
        if data.dtype.names is None:
            raise DataFormatError("Массив не является структурированным (нет полей)")
        if len(data) == 1:
            row = [data[name].item() for name in self.feature_names]
            X = np.array([row], dtype=np.float32)
            return X, False
        else:
            X = np.array([[row[name] for name in self.feature_names] for row in data], dtype=np.float32)
            return X, True

class PlainArrayProcessor(BaseProcessor):
    def process(self, data: Union[np.ndarray, List]):
        arr = np.asarray(data)
        if arr.ndim == 1:
            if len(arr) != len(self.feature_names):
                raise DataFormatError(f"Ожидается {len(self.feature_names)} признаков, получено {len(arr)}")
            X = np.array([arr], dtype=np.float32)
            return X, False
        elif arr.ndim == 2:
            if arr.shape[1] != len(self.feature_names):
                raise DataFormatError(f"Ожидается {len(self.feature_names)} столбцов, получено {arr.shape[1]}")
            return arr.astype(np.float32), True
        else:
            raise DataFormatError("Массив должен быть 1D или 2D")



def get_processor(data, feature_names):
    """
    Возвращает экземпляр подходящего процессора для данных.
    """
    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, (list, np.ndarray)):
            return DictBatchProcessor(feature_names)
        else:
            return DictSingleProcessor(feature_names)
    elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return ListOfDictsProcessor(feature_names)
    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        return StructuredArrayProcessor(feature_names)
    else:
        return PlainArrayProcessor(feature_names)