import pytest
import numpy as np
from unipredict import UniPredictor


def test_sklearn_predict(data_reg, sklearn_model):
    """Тест UniPredictor со sklearn моделью."""
    X, _, feature_names = data_reg
    predictor = UniPredictor(model=sklearn_model, feature_names=feature_names)
    
    # Одиночный словарь
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = predictor.predict(sample)
    assert result.ndim == 0 or (result.ndim == 1 and len(result) == 1)
    
    # Батч словарь
    batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
    result = predictor.predict(batch)
    assert result.shape == (5,)
    
    # Массив
    result = predictor.predict(X[:5])
    assert result.shape == (5,)

def test_torch_predict(data_reg, torch_model, norm_params):
    """Тест UniPredictor с PyTorch моделью."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    predictor = UniPredictor(
        model=torch_model,
        feature_names=feature_names,
        mean=mean,
        std=std
    )
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = predictor.predict(sample)
    assert result.ndim == 0 or (result.ndim == 1 and len(result) == 1)
    
    batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
    result = predictor.predict(batch)
    assert result.shape == (5, 1)

def test_tf_predict(data_reg, tf_model, norm_params):
    """Тест UniPredictor с TensorFlow моделью."""
    X, _, feature_names = data_reg
    mean, std = norm_params
    predictor = UniPredictor(
        model=tf_model,
        feature_names=feature_names,
        mean=mean,
        std=std
    )
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = predictor.predict(sample)
    assert result.ndim == 0 or (result.ndim == 1 and len(result) == 1)

def test_no_normalization(data_reg, sklearn_model):
    """Тест без нормализации."""
    X, _, feature_names = data_reg
    predictor = UniPredictor(model=sklearn_model, feature_names=feature_names)
    assert predictor.mean is None
    assert predictor.std is None
    sample = {name: X[0, i] for i, name in enumerate(feature_names)}
    result = predictor.predict(sample)
    assert result is not None

def test_normalization_partial(data_reg, sklearn_model):
    """Тест: только mean без std -> ошибка."""
    X, _, feature_names = data_reg
    mean = X.mean(axis=0)
    with pytest.raises(Exception):  # ConfigError
        predictor = UniPredictor(
            model=sklearn_model,
            feature_names=feature_names,
            mean=mean
        )

def test_predict_all_formats(data_reg, sklearn_model):
    """Тест всех форматов ввода."""
    X, _, feature_names = data_reg
    predictor = UniPredictor(model=sklearn_model, feature_names=feature_names)
    
    # 1. Одиночный словарь
    s = {name: X[0, i] for i, name in enumerate(feature_names)}
    r1 = predictor.predict(s)
    
    # 2. Словарь массивов
    b = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
    r2 = predictor.predict(b)
    
    # 3. Список словарей
    lst = [{name: X[i, j] for j, name in enumerate(feature_names)} for i in range(5)]
    r3 = predictor.predict(lst)
    
    # 4. Массив
    r4 = predictor.predict(X[:5])
    
    assert all(len(r) == 5 for r in [r2, r3, r4])