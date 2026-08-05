# Framework integrations

FlexPredict auto-detects built-in engines without importing optional frameworks during a base
package import.

## NumPy and custom Python models

The generic engine accepts any object with `predict(X)`. It passes DataFrames through and
also supports `predict_proba(X)` when the model provides it.

```python
class Model:
    def predict(self, values):
        ...


predictor = Predictor(Model(), task="regression")
```

Set `_estimator_type` to `"regressor"` or `"classifier"`, or pass `task` explicitly.

## scikit-learn

Install `flexpredict[sklearn]`, and add `pandas` when DataFrames are used:

```bash
python -m pip install "flexpredict[sklearn,pandas]"
```

The generic engine preserves DataFrame names, ordering and dtypes. FlexPredict recognizes
modern sklearn tags and the legacy `_estimator_type`, reads `classes_`, and uses
`predict_proba()` when requested.

Complete `.joblib`, `.pkl` and `.pickle` artifacts can be loaded through
{meth}`flexpredict.Predictor.from_file`.

## PyTorch

```bash
python -m pip install "flexpredict[torch]"
```

{class}`~flexpredict.Predictor` detects subclasses of `torch.nn.Module`. The engine:

- moves the model to the configured device;
- switches it to evaluation mode;
- creates tensors using the configured dtype and device;
- executes under `torch.inference_mode()`;
- detaches, moves to CPU and converts output tensors to NumPy.

```python
predictor = Predictor(
    torch_model,
    features=["x1", "x2"],
    task="regression",
    engine_options={"device": "cuda", "dtype": "float32"},
)
```

The default device is `cpu` and the default dtype is `float32`. Invalid device, dtype or
engine options fail at configuration time.

PyTorch modules do not have a standard probability API. The built-in engine accepts a model
`predict_proba()` method; otherwise use a probability-producing adapter or configure the
regular forward output as probabilities.

## TensorFlow and Keras

```bash
python -m pip install "flexpredict[tensorflow]"
```

Subclasses of `tf.keras.Model` are auto-detected. Input is converted to NumPy and inference
calls `model.predict(values, verbose=0)`.

```python
predictor = Predictor(
    keras_model,
    features=["x1", "x2"],
    task="regression",
)
```

Complete `.keras`, `.h5` and `.hdf5` artifacts are supported. Use `loader_options`, for
example `{"compile": False}`, when loading a saved Keras model.

## Heterogeneous ensembles

Frameworks do not need to match. A PyTorch predictor, an sklearn pipeline and a custom NumPy
model may be combined if their final result metadata and shapes are compatible. Each member
can select a different feature subset and order from the same request.
