# Loading model artifacts

## Complete sklearn/joblib artifacts

```python
from flexpredict import Predictor

predictor = Predictor.from_file(
    "models/random_forest.joblib",
    features=["age", "income", "score"],
)
```

Extensions `.joblib`, `.pkl` and `.pickle` select the joblib loader. Additional keyword
arguments are forwarded through `loader_options`.

## Complete Keras artifacts

```python
predictor = Predictor.from_file(
    "models/network.keras",
    features=["age", "income", "score"],
    task="classification",
    output_kind="probabilities",
    loader_options={"compile": False},
)
```

Extensions `.keras`, `.h5` and `.hdf5` select `tf.keras.models.load_model`.

## PyTorch state dictionaries

A state dict does not describe the Python architecture. Supply a factory:

```python
predictor = Predictor.from_torch_weights(
    "models/network_weights.pth",
    model_factory=lambda: MyNetwork(input_size=3),
    features=["age", "income", "score"],
    task="regression",
    engine_options={"device": "cuda", "dtype": "float32"},
)
```

The checkpoint is loaded with `weights_only=True` and defaults to the predictor device as
`map_location`. `strict` is forwarded to `load_state_dict`. If the checkpoint is a mapping
with a `state_dict` entry, that entry is selected automatically. Use `state_dict_key` for a
different nested key and `load_options` for `torch.load` options.

## Complete PyTorch models

`.pt` and `.pth` are intentionally never auto-detected: either extension may contain a state
dict, a complete model or an application-specific checkpoint. Loading a trusted complete
model must be explicit:

```python
predictor = Predictor.from_file(
    "models/trusted_model.pt",
    loader="torch_model",
    task="regression",
)
```

This path defaults to `weights_only=False` and therefore can execute code during
deserialization. Prefer state dictionaries when possible.

## Explicit loaders and unknown extensions

Use {func}`flexpredict.load_model` when loading and predictor construction should be separate:

```python
from flexpredict import Predictor, load_model

model = load_model("model.bin", loader="joblib")
predictor = Predictor(model)
```

Files must exist and be regular files. When `loader="auto"`, unknown extensions produce a
configuration error instead of guessing.

See [Security](../security.md) before loading artifacts from outside your application build
pipeline.
