"""FlexPredict public API."""

from .engines import InferenceEngine, register_engine
from .ensemble import EnsemblePredictor
from .exceptions import (
    ConfigurationError,
    EngineNotAvailableError,
    EnsembleCompatibilityError,
    FlexPredictError,
    InferenceError,
    InputValidationError,
    MissingFeatureError,
    OutputValidationError,
    PreprocessingError,
    UnexpectedFeatureError,
    UnsupportedOutputError,
)
from .predictor import Predictor
from .preprocessing import Standardizer
from .result import PredictionResult
from .schema import MISSING, FeatureSpec, InputSchema

__all__ = [
    "MISSING",
    "ConfigurationError",
    "EngineNotAvailableError",
    "EnsembleCompatibilityError",
    "EnsemblePredictor",
    "FeatureSpec",
    "FlexPredictError",
    "InferenceEngine",
    "InferenceError",
    "InputSchema",
    "InputValidationError",
    "MissingFeatureError",
    "OutputValidationError",
    "PredictionResult",
    "Predictor",
    "PreprocessingError",
    "Standardizer",
    "UnexpectedFeatureError",
    "UnsupportedOutputError",
    "register_engine",
]

__version__ = "0.2.0"

