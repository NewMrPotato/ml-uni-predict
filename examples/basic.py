"""Zero-configuration and named-input FlexPredict examples."""

import numpy as np

from flexpredict import Predictor


class SumRegressor:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1)


array_predictor = Predictor(SumRegressor())
print(array_predictor.predict([[1, 2], [3, 4]]).values)

named_predictor = Predictor(SumRegressor(), features=["x1", "x2"])
print(named_predictor.predict({"x1": 1, "x2": 2}).single())
print(named_predictor.predict({"x1": [1, 3], "x2": [2, 4]}).values)

