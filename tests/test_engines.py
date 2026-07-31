import sys
import types

import numpy as np
import pytest

import flexpredict
from flexpredict import (
    ConfigurationError,
    EngineNotAvailableError,
    InferenceError,
    OutputValidationError,
    Predictor,
    UnsupportedOutputError,
    register_engine,
)
from flexpredict.engines import (
    GenericEngine,
    TensorFlowEngine,
    TorchEngine,
    create_engine,
    normalize_output,
)


class Model:
    def predict(self, values):
        return np.ones(len(values))


def test_base_import_does_not_import_optional_frameworks():
    assert flexpredict.__version__ == "0.2.0"
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_auto_detection_uses_generic_engine_for_predict_models():
    assert isinstance(create_engine(Model()), GenericEngine)


def test_modern_sklearn_tags_drive_metadata_without_legacy_attribute():
    class Tags:
        estimator_type = "classifier"

    class TaggedClassifier:
        classes_ = np.array(["negative", "positive"])

        def __sklearn_tags__(self):
            return Tags()

        def predict(self, values):
            return np.full(len(values), "positive")

        def predict_proba(self, values):
            return np.tile([0.25, 0.75], (len(values), 1))

    predictor = Predictor(TaggedClassifier())
    result = predictor.predict_proba([[1], [2]])

    assert predictor.task == "classification"
    assert result.classes.tolist() == ["negative", "positive"]


def test_unknown_engine_has_clear_error():
    with pytest.raises(ConfigurationError, match="Unknown engine"):
        Predictor(Model(), engine="missing")


def test_generic_engine_rejects_unknown_options():
    with pytest.raises(ConfigurationError, match="Invalid options"):
        Predictor(Model(), engine_options={"unknown": True})


def test_output_normalization_distinguishes_batch_and_multioutput():
    assert normalize_output([1, 2], n_samples=2).shape == (2, 1)
    assert normalize_output([1, 2], n_samples=1).shape == (1, 2)


@pytest.mark.parametrize(
    ("output", "samples", "message"),
    [
        (1.0, 2, "scalar"),
        ([1, 2, 3], 2, "length"),
        (np.ones((3, 1)), 2, "returned 3 samples"),
        (np.ones((2, 1, 1)), 2, "scalar, 1D or 2D"),
        (np.array([np.nan, 1.0]), 2, "NaN"),
    ],
)
def test_output_normalization_rejects_invalid_outputs(output, samples, message):
    with pytest.raises(OutputValidationError, match=message):
        normalize_output(output, n_samples=samples)


def test_generic_engine_wraps_model_errors_and_missing_probability_method():
    class Broken:
        def predict(self, values):
            raise RuntimeError("boom")

    engine = GenericEngine(Broken())
    with pytest.raises(InferenceError, match="boom"):
        engine.predict(np.ones((1, 1)))
    with pytest.raises(UnsupportedOutputError, match="predict_proba"):
        engine.predict_proba(np.ones((1, 1)))


def test_output_selector_handles_named_outputs():
    class MultiOutput:
        def predict(self, values):
            return {"first": np.zeros((len(values), 1)), "second": np.ones((len(values), 1))}

    selected = Predictor(MultiOutput(), output_selector="second").predict([[1]])
    assert selected.single() == 1.0

    with pytest.raises(OutputValidationError, match="multiple named outputs"):
        Predictor(MultiOutput()).predict([[1]])


def test_custom_engine_registration_and_detection():
    class SpecialModel:
        pass

    class SpecialEngine(GenericEngine):
        def predict(self, values):
            return np.full(len(values), 7)

    register_engine(
        "test-special",
        SpecialEngine,
        detector=lambda model: isinstance(model, SpecialModel),
        priority=100,
    )

    assert Predictor(SpecialModel()).predict([[1]]).single() == 7
    with pytest.raises(ConfigurationError, match="already registered"):
        register_engine("test-special", SpecialEngine)


def test_fake_torch_engine_is_lazy_and_supports_user_subclasses(monkeypatch):
    module = types.ModuleType("torch")

    class Module:
        __module__ = "torch.nn"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    module.nn = types.SimpleNamespace(Module=Module)
    module.float32 = "float32"
    module.device = lambda value: value
    module.as_tensor = lambda values, **kwargs: Tensor(values)
    module.inference_mode = InferenceMode
    monkeypatch.setitem(sys.modules, "torch", module)

    class UserModel(Module):
        def __call__(self, tensor):
            return Tensor(tensor.values.sum(axis=1))

    predictor = Predictor(UserModel(), task="regression")
    result = predictor.predict([[1, 2], [3, 4]])

    assert isinstance(predictor._engine, TorchEngine)
    assert np.allclose(result.values, [[3], [7]])


def test_fake_tensorflow_engine(monkeypatch):
    module = types.ModuleType("tensorflow")

    class KerasModel:
        __module__ = "tensorflow.keras"

    module.keras = types.SimpleNamespace(Model=KerasModel)
    monkeypatch.setitem(sys.modules, "tensorflow", module)

    class UserModel(KerasModel):
        def predict(self, values, verbose=0):
            return np.asarray(values).sum(axis=1)

    predictor = Predictor(UserModel(), task="regression")

    assert isinstance(predictor._engine, TensorFlowEngine)
    assert predictor.predict([[1, 2]]).single() == 3


def test_missing_optional_engine_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    class TorchBase:
        __module__ = "torch.nn"

    class UserModel(TorchBase):
        pass

    with pytest.raises(EngineNotAvailableError, match=r"flexpredict\[torch\]"):
        Predictor(UserModel())
