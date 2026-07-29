# UniPredict: Universal ML Model Predictor

![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)
![versions](https://img.shields.io/badge/python-3.7%2B-blue)


## Table of Contents
- [Overview and Motivation](#overview-and-motivation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Code Examples](#code-examples)
- [Ensemble Support](#ensemble-support)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Why UniPredict?](#why-unipredict)
- [License](#license)

## Overview and Motivation

During a machine learning competition, I faced a common yet frustrating challenge: integrating a complex ensemble model into an existing application. The model was built using multiple frameworks—scikit-learn, PyTorch, and XGBoost—each requiring different data formats and preprocessing steps. Meanwhile, the integration target demanded flexibility in input formats: sometimes a dictionary, sometimes a numpy array etc.

The repeated code for data transformation, model prediction, and normalization quickly became unwieldy. Each new model required writing the same boilerplate code over and over. I realized there had to be a better way.

This is how `UniPredict` was born—a universal wrapper that simplifies model inference by handling input data in any format and supporting models from any framework (almost). Whether you're working with a single model or a complex ensemble, `UniPredict` provides a consistent, clean interface that "just works."

## Key Features

- **Universal Model Support**: Works with scikit-learn, PyTorch, TensorFlow/Keras, and any custom model with a `predict` method.
- **Flexible Input Formats**: Accepts data as dictionaries, lists of dictionaries, structured arrays, or plain arrays—automatically converting them to the format your model expects.
- **Built-in Normalization**: Handles data standardization with `mean` and `std` parameters, eliminating extra preprocessing steps.
- **Ensemble Support**: Combine multiple predictors with strategies like weighted mean, median, max, or custom aggregation functions.
- **Intelligent Engine Detection**: Automatically detects the framework (PyTorch, TensorFlow, scikit-learn) and selects the appropriate inference engine.
- **Zero Boilerplate**: Create a predictor in one line and start making predictions immediately.

## Installation

<!-- ### Basic Installation
```bash
pip install unipredict
```

### With Optional Dependencies
```bash
pip install unipredict[torch]      # For PyTorch support
pip install unipredict[tensorflow] # For TensorFlow support
pip install unipredict[all]        # All dependencies
``` -->

### Development Installation
```bash
git clone https://github.com/NewMrPotato/ml-uni-predict.git
cd ml-uni-predict
pip install -e .
```

## Quick Start

```python
from unipredict import UniPredictor
import numpy as np

# Create a predictor for any model
predictor = UniPredictor(
    model=my_model,                    # sklearn, PyTorch, or TensorFlow
    feature_names=['feat1', 'feat2']   # Required for flexible input formats
)

# Predict from a dictionary (single object)
result = predictor.predict({'feat1': 1.0, 'feat2': 2.0})

# Predict from a batch dictionary
batch = {'feat1': [1.0, 2.0], 'feat2': [3.0, 4.0]}
results = predictor.predict(batch)

# Predict from a numpy array
X = np.array([[1.0, 2.0], [3.0, 4.0]])
results = predictor.predict(X)
```

## Code Examples

Full working examples for each framework are available in the [`examples/`](examples/) directory.

### PyTorch Example (with normalization)
```python
predictor = UniPredictor(
    model=torch_model,
    feature_names=['x1', 'x2', 'x3'],
    mean=mean_values,
    std=std_values,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Works with any input format
result = predictor.predict({'x1': 1.2, 'x2': 2.3, 'x3': 0.5})
result = predictor.predict(X_numpy_array)
result = predictor.predict(list_of_dicts)
```

### Scikit-learn, TensorFlow, and Custom Models
Check out the dedicated examples:
- [Scikit-learn example](examples/example_sklearn.py)
- [TensorFlow example](examples/example_tensorflow.py)
- [Custom model example](examples/example_custom_model.py)
- [Cascade predictor example](examples/example_cascade.py)

## Ensemble Support

`UniPredict` provides flexible ensemble capabilities through `EnsemblePredictor`, supporting various aggregation strategies.

### Weighted Ensemble
```python
from unipredict import EnsemblePredictor

ensemble = EnsemblePredictor(
    [predictor1, predictor2, predictor3],
    weights=[0.5, 0.3, 0.2],
    aggregation='weighted_mean'
)

# Same flexible input format as UniPredictor
result = ensemble.predict({'feat1': 1.0, 'feat2': 2.0})
```

### Aggregation Strategies
- `'weighted_mean'` — Weighted average with custom weights
- `'mean'` — Simple average
- `'median'` — Median (robust to outliers)
- `'max'` — Maximum value
- `'min'` — Minimum value
- Custom function — Your own aggregation logic

### Custom Aggregation Example
```python
def geometric_mean(predictions, epsilon=1e-8):
    log_preds = np.log(np.abs(predictions) + epsilon)
    return np.exp(np.mean(log_preds, axis=0))

ensemble = EnsemblePredictor(
    [predictor1, predictor2, predictor3],
    aggregation=geometric_mean
)
```

## API Reference

### UniPredictor
```python
UniPredictor(
    model: Any,                          # Trained model
    config: Optional[ModelConfig] = None, # Configuration object
    **kwargs                             # Optional: feature_names, mean, std, device
)
```

**Key Parameters:**
- `feature_names` (required): List of feature names
- `mean` (optional): Array of mean values for normalization
- `std` (optional): Array of standard deviation values for normalization
- `device` (optional): 'cpu' or 'cuda' (for PyTorch models)

**Supported Input Formats:**
- Dictionary (single object): `{'feat1': 1.0, 'feat2': 2.0}`
- Dictionary of arrays (batch): `{'feat1': [1.0, 2.0], 'feat2': [3.0, 4.0]}`
- List of dictionaries: `[{'feat1': 1.0, 'feat2': 2.0}, ...]`
- Structured numpy array: `np.array([(1.0, 2.0)], dtype=[('feat1', float), ('feat2', float)])`
- Plain numpy array: `np.array([[1.0, 2.0], [3.0, 4.0]])`

### EnsemblePredictor
```python
EnsemblePredictor(
    predictors: List[Callable],          # List of predictors
    weights: Optional[List[float]] = None,# Optional weights
    aggregation: Union[str, Callable] = 'weighted_mean',
    normalize_weights: bool = True
)
```

## Testing

The project uses `pytest` for comprehensive testing. Tests cover all core components:

- **Processors**: Input data conversion for all supported formats.
- **Engines**: Inference engines for sklearn, PyTorch, and TensorFlow.
- **Core**: Full `UniPredictor` functionality with normalization and format handling.
- **Ensemble**: All aggregation strategies, custom functions, and edge cases.

### Running Tests Locally

1. **Generate test models** (required before running tests):
   ```bash
   python generate_test_models.py
   ```
   This creates dummy models for sklearn, PyTorch, and TensorFlow in the `test_models/` directory.

2. **Install development dependencies** (if not already installed):
   ```bash
   pip install pytest pytest-cov
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **With coverage report**:
   ```bash
   pytest tests/ --cov=unipredict --cov-report=html
   ```


## Why UniPredict?

| Feature | UniPredict | ml-wrappers | Manual Approach |
|---------|------------|-------------|-----------------|
| **Input Flexibility** | ✅ Dictionary, array, list | ✅ DatasetWrapper | ❌ Custom code needed |
| **Multi-Framework** | ✅ sklearn, PyTorch, TF | ✅ Many frameworks | ❌ Inconsistent APIs |
| **Ensemble Support** | ✅ Built-in | ❌ Not supported | ❌ Complex implementation |
| **Normalization** | ✅ Built-in | ❌ Manual | ❌ Manual |
| **One-Line Setup** | ✅ Yes | ❌ Multiple steps | ❌ Multiple steps |
| **Zero Boilerplate** | ✅ Yes | ⚠️ Some needed | ❌ Much needed |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for data scientists and ML engineers**
