
class UniPredictError(Exception):
    """Базовое исключение для пакета."""
    pass

class DataFormatError(UniPredictError):
    """Неподдерживаемый формат входных данных."""
    pass

class FeatureNotFoundError(UniPredictError):
    """Отсутствует необходимый признак."""
    pass

class ModelLoadingError(UniPredictError):
    """Ошибка при загрузке модели."""
    pass

class ConfigError(UniPredictError):
    """Ошибка в конфигурации."""
    pass