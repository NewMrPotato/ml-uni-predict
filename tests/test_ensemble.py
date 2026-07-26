import pytest
import numpy as np
from unipredict import UniPredictor, EnsemblePredictor


def test_weighted_ensemble(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Взвешенное среднее."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    ensemble = EnsemblePredictor([p1, p2, p3], weights=[0.5, 0.3, 0.2])
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = ensemble.predict(sample)
    assert result.ndim == 0 or (result.ndim == 1 and len(result) == 1)

def test_mean_ensemble(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Простое среднее."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    ensemble = EnsemblePredictor([p1, p2, p3], aggregation='mean')
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = ensemble.predict(sample)
    assert result is not None

def test_median_ensemble(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Медиана."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    ensemble = EnsemblePredictor([p1, p2, p3], aggregation='median')
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = ensemble.predict(sample)
    assert result is not None

def test_max_ensemble(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Максимум."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    ensemble = EnsemblePredictor([p1, p2, p3], aggregation='max')
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = ensemble.predict(sample)
    assert result is not None

def test_custom_function_ensemble(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Пользовательская функция агрегации."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    def geometric_mean(preds, epsilon=1e-8):
        log_preds = np.log(np.abs(preds) + epsilon)
        return np.exp(np.mean(log_preds, axis=0))
    
    ensemble = EnsemblePredictor([p1, p2, p3], aggregation=geometric_mean)
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = ensemble.predict(sample)
    assert result is not None

def test_ensemble_batch(data_reg, sklearn_model, torch_model, tf_model, norm_params):
    """Ансамбль с батчем."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=torch_model, feature_names=feature_names, mean=mean, std=std)
    p3 = UniPredictor(model=tf_model, feature_names=feature_names, mean=mean, std=std)
    
    ensemble = EnsemblePredictor([p1, p2, p3], aggregation='mean')
    batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
    result = ensemble.predict(batch)
    assert result.shape == (5, 1)

def test_ensemble_empty():
    """Пустой ансамбль -> ошибка."""
    with pytest.raises(ValueError):
        EnsemblePredictor([])

def test_ensemble_wrong_weights(data_reg, sklearn_model):
    """Неправильное число весов."""
    X, _, feature_names = data_reg
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    p2 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    with pytest.raises(ValueError):
        EnsemblePredictor([p1, p2], weights=[0.5])

def test_ensemble_unknown_aggregation(data_reg, sklearn_model):
    """Неизвестная стратегия агрегации."""
    X, _, feature_names = data_reg
    p1 = UniPredictor(model=sklearn_model, feature_names=feature_names)
    with pytest.raises(ValueError):
        EnsemblePredictor([p1], aggregation='unknown')