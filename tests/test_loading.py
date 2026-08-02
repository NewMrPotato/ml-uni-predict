import sys
import types

import numpy as np
import pytest

from flexpredict import ConfigurationError, Predictor, load_model, load_torch_state_dict


class LoadedRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.ones(len(values))


def test_predictor_from_file_uses_lazy_joblib_loader(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"test")
    module = types.ModuleType("joblib")
    module.load = lambda path, **options: LoadedRegressor()
    monkeypatch.setitem(sys.modules, "joblib", module)

    predictor = Predictor.from_file(artifact)

    assert predictor.predict([[1, 2]]).single() == 1.0


def test_tensorflow_loader_is_lazy_and_receives_options(tmp_path, monkeypatch):
    artifact = tmp_path / "model.keras"
    artifact.write_bytes(b"test")
    calls = {}
    module = types.ModuleType("tensorflow")

    def fake_load(path, **options):
        calls.update(path=path, options=options)
        return LoadedRegressor()

    module.keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=fake_load))
    monkeypatch.setitem(sys.modules, "tensorflow", module)

    model = load_model(artifact, loader_options={"compile": False})

    assert isinstance(model, LoadedRegressor)
    assert calls["options"] == {"compile": False}


def test_torch_state_dict_loader_uses_safe_defaults(tmp_path, monkeypatch):
    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"test")
    calls = {}
    module = types.ModuleType("torch")

    def fake_load(path, **options):
        calls.update(path=path, options=options)
        return {"state_dict": {"weight": 42}}

    module.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", module)

    class Model:
        def load_state_dict(self, state, *, strict):
            self.state = state
            self.strict = strict

    model = load_torch_state_dict(Model, artifact)

    assert model.state == {"weight": 42}
    assert model.strict is True
    assert calls["options"] == {"weights_only": True, "map_location": "cpu"}


def test_torch_state_dict_key_is_validated(tmp_path, monkeypatch):
    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"test")
    module = types.ModuleType("torch")
    module.load = lambda path, **options: {"weights": {}}
    monkeypatch.setitem(sys.modules, "torch", module)

    with pytest.raises(ConfigurationError, match="no state dict key"):
        load_torch_state_dict(lambda: object(), artifact, state_dict_key="missing")


def test_complete_torch_model_loader_is_explicit(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"test")
    module = types.ModuleType("torch")
    loaded = LoadedRegressor()
    calls = {}

    def fake_load(path, **options):
        calls.update(options)
        return loaded

    module.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", module)

    assert load_model(artifact, loader="torch_model") is loaded
    assert calls == {"weights_only": False}


def test_torch_state_dict_validates_factory_and_model(tmp_path, monkeypatch):
    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"test")
    module = types.ModuleType("torch")
    module.load = lambda path, **options: {}
    monkeypatch.setitem(sys.modules, "torch", module)

    with pytest.raises(ConfigurationError, match="must be callable"):
        load_torch_state_dict(None, artifact)
    with pytest.raises(ConfigurationError, match="model_factory failed"):
        load_torch_state_dict(lambda: 1 / 0, artifact)
    with pytest.raises(ConfigurationError, match="load_state_dict"):
        load_torch_state_dict(object, artifact)


def test_unknown_explicit_loader_is_rejected(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"test")

    with pytest.raises(ConfigurationError, match="Unknown model loader"):
        load_model(artifact, loader="unknown")


def test_auto_loader_rejects_ambiguous_pytorch_file(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"test")

    with pytest.raises(ConfigurationError, match="ambiguous"):
        load_model(artifact)


def test_loader_rejects_missing_and_unknown_files(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_model(tmp_path / "missing.joblib")

    artifact = tmp_path / "model.unknown"
    artifact.write_bytes(b"test")
    with pytest.raises(ConfigurationError, match="Cannot infer"):
        load_model(artifact)
