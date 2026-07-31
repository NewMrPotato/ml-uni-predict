import sys

import numpy as np
import pytest

import flexpredict
from flexpredict import ConfigurationError, Predictor
from flexpredict.engines import GenericEngine, create_engine, normalize_output


class Model:
    def predict(self, values):
        return np.ones(len(values))


def test_base_import_does_not_import_optional_frameworks():
    assert flexpredict.__version__ == "0.2.0"
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_auto_detection_uses_generic_engine_for_predict_models():
    assert isinstance(create_engine(Model()), GenericEngine)


def test_unknown_engine_has_clear_error():
    with pytest.raises(ConfigurationError, match="Unknown engine"):
        Predictor(Model(), engine="missing")


def test_generic_engine_rejects_unknown_options():
    with pytest.raises(ConfigurationError, match="Invalid options"):
        Predictor(Model(), engine_options={"unknown": True})


def test_output_normalization_distinguishes_batch_and_multioutput():
    assert normalize_output([1, 2], n_samples=2).shape == (2, 1)
    assert normalize_output([1, 2], n_samples=1).shape == (1, 2)
