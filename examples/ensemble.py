"""Heterogeneous-style ensemble with global weights from a NumPy file."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from flexpredict import EnsemblePredictor, Predictor


class ConstantRegressor:
    _estimator_type = "regressor"

    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return np.full(len(values), self.value, dtype=float)


predictors = [
    Predictor(ConstantRegressor(100), name="model_a"),
    Predictor(ConstantRegressor(120), name="model_b"),
    Predictor(ConstantRegressor(90), name="model_c"),
]

with TemporaryDirectory() as directory:
    weights_path = Path(directory) / "global_weights.npy"
    np.save(weights_path, np.array([0.5, 0.3, 0.2]))

    ensemble = EnsemblePredictor(
        predictors,
        aggregation="weighted_mean",
        aggregation_weights=weights_path,
    )
    print(ensemble.predict([[1, 2, 3]]).single())  # 104.0

