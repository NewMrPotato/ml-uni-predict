import pytest
import numpy as np
from unipredict.processors import (
    DictSingleProcessor, DictBatchProcessor, ListOfDictsProcessor,
    StructuredArrayProcessor, PlainArrayProcessor, get_processor
)
from unipredict.exceptions import DataFormatError, FeatureNotFoundError


def test_dict_single():
    """Одиночный словарь."""
    feature_names = ['x1', 'x2', 'x3']
    data = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0}
    proc = DictSingleProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (1, 3)
    assert not is_batch
    assert np.allclose(X, [[1.0, 2.0, 3.0]])

def test_dict_single_missing():
    """Одиночный словарь с отсутствующим признаком."""
    feature_names = ['x1', 'x2', 'x3']
    data = {'x1': 1.0, 'x2': 2.0}
    proc = DictSingleProcessor(feature_names)
    with pytest.raises(FeatureNotFoundError):
        proc.process(data)

def test_dict_batch():
    """Словарь массивов (батч)."""
    feature_names = ['x1', 'x2']
    data = {'x1': [1.0, 2.0, 3.0], 'x2': [4.0, 5.0, 6.0]}
    proc = DictBatchProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (3, 2)
    assert is_batch
    assert np.allclose(X, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

def test_dict_batch_wrong_length():
    """Словарь массивов разной длины."""
    feature_names = ['x1', 'x2']
    data = {'x1': [1.0, 2.0], 'x2': [4.0, 5.0, 6.0]}
    proc = DictBatchProcessor(feature_names)
    with pytest.raises(DataFormatError):
        proc.process(data)

def test_list_of_dicts():
    """Список словарей."""
    feature_names = ['x1', 'x2']
    data = [{'x1': 1.0, 'x2': 2.0}, {'x1': 3.0, 'x2': 4.0}]
    proc = ListOfDictsProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (2, 2)
    assert is_batch
    assert np.allclose(X, [[1.0, 2.0], [3.0, 4.0]])

def test_structured_array_1d():
    """Структурированный массив (1D)."""
    feature_names = ['x1', 'x2']
    dtype = [('x1', float), ('x2', float)]
    data = np.array([(1.0, 2.0)], dtype=dtype)
    proc = StructuredArrayProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (1, 2)
    assert not is_batch
    assert np.allclose(X, [[1.0, 2.0]])

def test_structured_array_2d():
    """Структурированный массив (2D)."""
    feature_names = ['x1', 'x2']
    dtype = [('x1', float), ('x2', float)]
    data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
    proc = StructuredArrayProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (2, 2)
    assert is_batch
    assert np.allclose(X, [[1.0, 2.0], [3.0, 4.0]])

def test_plain_array_1d():
    """Обычный 1D массив."""
    feature_names = ['x1', 'x2', 'x3']
    data = np.array([1.0, 2.0, 3.0])
    proc = PlainArrayProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (1, 3)
    assert not is_batch
    assert np.allclose(X, [[1.0, 2.0, 3.0]])

def test_plain_array_2d():
    """Обычный 2D массив."""
    feature_names = ['x1', 'x2']
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    proc = PlainArrayProcessor(feature_names)
    X, is_batch = proc.process(data)
    assert X.shape == (2, 2)
    assert is_batch
    assert np.allclose(X, [[1.0, 2.0], [3.0, 4.0]])

def test_plain_array_wrong_shape():
    """Обычный массив с неправильным числом признаков."""
    feature_names = ['x1', 'x2']
    data = np.array([1.0, 2.0, 3.0])
    proc = PlainArrayProcessor(feature_names)
    with pytest.raises(DataFormatError):
        proc.process(data)

def test_get_processor():
    """Проверка диспетчера."""

    feature_names = ['x1', 'x2']

    # Словарь
    data = {'x1': 1.0, 'x2': 2.0}
    proc = get_processor(data, feature_names)
    assert isinstance(proc, DictSingleProcessor)

    # Словарь массивов
    data = {'x1': [1.0, 2.0], 'x2': [3.0, 4.0]}
    proc = get_processor(data, feature_names)
    assert isinstance(proc, DictBatchProcessor)

    # Список словарей
    data = [{'x1': 1.0, 'x2': 2.0}]
    proc = get_processor(data, feature_names)
    assert isinstance(proc, ListOfDictsProcessor)

    # Структурированный массив
    dtype = [('x1', float), ('x2', float)]
    data = np.array([(1.0, 2.0)], dtype=dtype)
    proc = get_processor(data, feature_names)
    assert isinstance(proc, StructuredArrayProcessor)

    # Обычный массив
    data = np.array([1.0, 2.0])
    proc = get_processor(data, feature_names)
    assert isinstance(proc, PlainArrayProcessor)