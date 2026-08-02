import numpy as np
import pytest

from flexpredict import ConfigurationError, PreprocessingError, Standardizer
from flexpredict.preprocessing import apply_preprocessor


def test_standardizer_transforms_batch():
    transformer = Standardizer(mean=[1, 10], std=[2, 5])

    result = transformer.transform(np.array([[3, 20], [1, 5]]))

    assert np.allclose(result, [[1, 2], [0, -1]])


@pytest.mark.parametrize("std", [[1, 0], [1, -1], [1, np.nan]])
def test_standardizer_rejects_invalid_std(std):
    with pytest.raises(ConfigurationError):
        Standardizer(mean=[0, 0], std=std)


def test_callable_preprocessor_must_preserve_samples():
    with pytest.raises(PreprocessingError, match="preserve"):
        apply_preprocessor(lambda values: values[:1], np.ones((2, 2)))

