from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from .exceptions import ConfigError


@dataclass
class ModelConfig:
    """
    Настройки для UniPredictor.
    
    :param feature_names: список имён признаков (обязательно)
    :param mean: средние значения для нормализации (np.ndarray).
    :param std: стандартные отклонения для нормализации (np.ndarray).
    :param device: устройство для инференса ('cpu' или 'cuda'), используется только для PyTorch.
    :param engine_type: принудительный тип движка ('sklearn', 'torch', 'tensorflow').
                        Если не указан, определяется автоматически по типу модели.
    :param model_kwargs: дополнительные аргументы для движка (например, для TorchEngine).
    """
    feature_names: List[str]
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    device: str = "cpu"
    engine_type: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.feature_names:
            raise ConfigError("feature_names не может быть пустым")
        
        if self.mean is not None and self.std is not None:
            if len(self.mean) != len(self.feature_names):
                raise ConfigError("mean должно содержать столько же значений, сколько признаков")
            if len(self.std) != len(self.feature_names):
                raise ConfigError("std должно содержать столько же значений, сколько признаков")
        elif self.mean is not None or self.std is not None:
            raise ConfigError("mean и std должны быть заданы одновременно (оба None или оба не None)")