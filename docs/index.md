---
hide-toc: true
---

<div class="hero">

# One inference contract for every model

FlexPredict turns NumPy, scikit-learn, PyTorch, TensorFlow/Keras and custom models into
predictors with consistent inputs, validated outputs and safe heterogeneous ensembles.

```bash
python -m pip install flexpredict
```

</div>

FlexPredict is an inference composition library. It does not train models or run a model
server. Instead, it connects the parts that sit immediately around inference:

- array, record, column-oriented and DataFrame inputs;
- feature selection, ordering, validation and preprocessing;
- framework-specific execution and artifact loading;
- a canonical two-dimensional prediction result;
- regression and classification ensembles whose members may use different features and
  frameworks.

```python
import numpy as np

from flexpredict import Predictor


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


predictor = Predictor(SumRegressor(), features=["x1", "x2"])
result = predictor.predict({"x1": [1, 3], "x2": [2, 4]})

print(result.values)  # [[3.0], [7.0]]
```

:::{note}
Version 0.2 is currently an alpha redesign. It intentionally does not preserve the old
`unipredict` API. Pin the package version in production environments.
:::

## Choose a path

**New to FlexPredict?** Start with [Installation and quick start](getting-started.md), then
read [Core concepts](concepts.md).

**Connecting application data?** See [Input data and schemas](guides/input-data.md) and
[Preprocessing](guides/preprocessing.md).

**Integrating a framework or artifact?** See [Framework integrations](guides/frameworks.md),
[Loading model artifacts](guides/model-loading.md) and [Custom engines and outputs](guides/engines.md).

**Combining models?** See [Ensembles](guides/ensembles.md), including class alignment,
weighted aggregation and heterogeneous feature sets.

```{toctree}
:hidden:
:maxdepth: 2

getting-started
concepts
guides/input-data
guides/preprocessing
guides/classification
guides/frameworks
guides/model-loading
guides/engines
guides/ensembles
guides/errors
security
api/index
changelog
contributing
ru/index
```
