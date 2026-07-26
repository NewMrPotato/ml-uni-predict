import numpy as np
from typing import List, Optional, Callable


class EnsemblePredictor:
    """
    Класс для ансамбля моделей с гибкой стратегией агрегации.
    
    Поддерживает:
    - Взвешенное среднее (weighted_mean)
    - Простое среднее (mean)
    - Медиана (median)
    - Максимум (max)
    - Минимум (min)
    - Пользовательская функция агрегации
    """
    
    VALID_AGGREGATIONS = {"weighted_mean", "mean", "median", "max", "min"}
    
    def __init__(
        self,
        predictors: List[Callable],
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_mean",
        normalize_weights: bool = True,
    ):
        if not predictors:
            raise ValueError("Список предикторов не может быть пустым")
        
        self.predictors = predictors
        
        if isinstance(aggregation, str):
            if aggregation not in self.VALID_AGGREGATIONS:
                raise ValueError(f"Неизвестная стратегия агрегации: {aggregation}")
        self.aggregation = aggregation
        
        if weights is None:
            weights = [1.0] * len(predictors)
        elif len(weights) != len(predictors):
            raise ValueError("Количество весов должно совпадать с количеством предикторов")
        
        self.weights = np.array(weights, dtype=np.float32)
        if normalize_weights:
            self.weights = self.weights / self.weights.sum()
    
    def predict(self, data):
        preds = []
        for predictor in self.predictors:
            pred = predictor.predict(data)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)
        
        stacked = np.stack(preds, axis=0)  # (n_models, n_samples, n_outputs)
        
        if isinstance(self.aggregation, str):
            if self.aggregation == "weighted_mean":
                result = np.average(stacked, axis=0, weights=self.weights)
            elif self.aggregation == "mean":
                result = np.mean(stacked, axis=0)
            elif self.aggregation == "median":
                result = np.median(stacked, axis=0)
            elif self.aggregation == "max":
                result = np.max(stacked, axis=0)
            elif self.aggregation == "min":
                result = np.min(stacked, axis=0)
        else:
            result = self.aggregation(stacked)
        
        if result.ndim == 2 and result.shape[0] == 1:
            result = result.flatten()
        return result