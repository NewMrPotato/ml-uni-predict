# FlexPredict 0.2.0a1

This is the first alpha release of the redesigned FlexPredict inference composition layer.
It intentionally does not preserve the former `unipredict` API.

## Highlights

- One `Predictor` API for NumPy, sklearn-compatible, PyTorch, TensorFlow/Keras and custom
  models, with lazy optional framework imports.
- Named feature selection and explicit `InputSchema` validation for dictionaries, records,
  arrays, structured arrays and pandas inputs.
- Preserved DataFrame column names and dtypes for sklearn `Pipeline` and
  `ColumnTransformer` inference.
- Callable, transformer-based and built-in standardization preprocessing.
- Canonical two-dimensional `PredictionResult` values with explicit task, output kind and
  class metadata.
- Mean, weighted mean, median, min, max and voting ensembles, including safe class-order
  alignment and normalized global weights loaded from `.npy` files.
- Joblib, Keras, trusted full PyTorch and safe PyTorch state-dict loading paths.
- Strict validation for shapes, non-finite values, output dtypes and ensemble compatibility.

## Installation

After publication, install the base package with:

```bash
python -m pip install "flexpredict==0.2.0a1"
```

Install only the optional integrations needed by the application, for example:

```bash
python -m pip install "flexpredict[sklearn,pandas]==0.2.0a1"
```

Because this is an alpha release, public API changes may still occur before 1.0. Model
artifacts must be loaded only from trusted sources.
