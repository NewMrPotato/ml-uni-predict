"""Strict input validation and per-model preprocessing."""

import numpy as np

from flexpredict import FeatureSpec, InputSchema, Predictor, Standardizer


class RiskModel:
    _estimator_type = "regressor"

    def predict(self, values):
        return np.asarray(values).sum(axis=1)


schema = InputSchema(
    (
        FeatureSpec("age", int, validators=(lambda value: 18 <= value <= 120,)),
        FeatureSpec("income", float, validators=(lambda value: value >= 0,)),
        FeatureSpec("score", float, default=0.5),
    )
)

predictor = Predictor(
    RiskModel(),
    schema=schema,
    preprocessor=Standardizer(
        mean=[35, 100_000, 0.5],
        std=[12, 50_000, 0.2],
    ),
)

result = predictor.predict({"age": "31", "income": 150_000})
print(result.single())

