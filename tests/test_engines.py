import pytest
import numpy as np
import torch
import tensorflow as tf
from unipredict.engines import SklearnEngine, TorchEngine, TensorFlowEngine, get_engine


class TestSklearnEngine:
    """Тесты SklearnEngine."""

    def test_predict_shape(self, sklearn_model, X_sample):
        engine = SklearnEngine(sklearn_model)
        result = engine.predict(X_sample)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == X_sample.shape[0]
        assert result.ndim == 1 or result.shape[1] == 1

    def test_predict_single(self, sklearn_model):
        engine = SklearnEngine(sklearn_model)
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = engine.predict(X)
        assert result.shape == (1,)


class TestTorchEngine:
    """Тесты TorchEngine."""

    def test_predict_shape(self, torch_model, X_sample):
        engine = TorchEngine(torch_model, device='cpu')
        result = engine.predict(X_sample)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == X_sample.shape[0]
        assert result.ndim == 2 and result.shape[1] == 1

    def test_predict_single(self, torch_model):
        engine = TorchEngine(torch_model, device='cpu')
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = engine.predict(X)
        assert result.shape == (1, 1)

    def test_device(self, torch_model):
        engine = TorchEngine(torch_model, device='cpu')
        assert engine.device.type == 'cpu'


class TestTensorFlowEngine:
    """Тесты TensorFlowEngine."""

    def test_predict_shape(self, tf_model, X_sample):
        engine = TensorFlowEngine(tf_model)
        result = engine.predict(X_sample)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == X_sample.shape[0]
        assert result.ndim == 2 and result.shape[1] == 1

    def test_predict_single(self, tf_model):
        engine = TensorFlowEngine(tf_model)
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = engine.predict(X)
        assert result.shape == (1, 1)


class TestGetEngine:
    """Тесты фабрики движков."""

    def test_sklearn_engine(self, sklearn_model):
        engine = get_engine(sklearn_model)
        assert isinstance(engine, SklearnEngine)

    def test_torch_engine(self, torch_model):
        engine = get_engine(torch_model, device='cpu')
        assert isinstance(engine, TorchEngine)

    def test_tf_engine(self, tf_model):
        engine = get_engine(tf_model)
        assert isinstance(engine, TensorFlowEngine)

    def test_explicit_engine_type(self, sklearn_model):
        engine = get_engine(sklearn_model, engine_type='sklearn')
        assert isinstance(engine, SklearnEngine)

    def test_unknown_engine_type(self, sklearn_model):
        with pytest.raises(ValueError, match="Неизвестный тип движка"):
            get_engine(sklearn_model, engine_type='unknown')

    def test_custom_model_with_predict(self):
        """Модель с методом predict должна быть обёрнута в SklearnEngine."""
        class CustomModel:
            def predict(self, X):
                return np.ones(X.shape[0])
        model = CustomModel()
        engine = get_engine(model)
        assert isinstance(engine, SklearnEngine)

        X = np.array([[1.0, 2.0, 3.0]])
        result = engine.predict(X)
        assert np.allclose(result, np.ones(3))

    def test_custom_model_without_predict(self):
        """Модель без метода predict должна выбрасывать понятное исключение."""
        class CustomModel:
            pass
        model = CustomModel()
        with pytest.raises(ValueError, match="Не удалось определить тип модели"):
            get_engine(model)