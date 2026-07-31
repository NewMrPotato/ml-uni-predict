"""Public exception hierarchy for FlexPredict."""


class FlexPredictError(Exception):
    """Base class for all library-specific errors."""


class ConfigurationError(FlexPredictError):
    """The predictor or ensemble configuration is invalid."""


class InputValidationError(FlexPredictError):
    """Input data does not satisfy the declared input contract."""


class MissingFeatureError(InputValidationError):
    """A required feature is missing."""


class UnexpectedFeatureError(InputValidationError):
    """Input data contains a feature forbidden by the schema."""


class PreprocessingError(FlexPredictError):
    """A preprocessor failed or returned invalid data."""


class EngineNotAvailableError(FlexPredictError):
    """A requested optional inference engine is not installed."""


class InferenceError(FlexPredictError):
    """A model failed during inference."""


class OutputValidationError(FlexPredictError):
    """A model or aggregator returned an invalid output."""


class UnsupportedOutputError(FlexPredictError):
    """The model cannot produce the requested output kind."""


class EnsembleCompatibilityError(FlexPredictError):
    """Predictions cannot be combined safely."""

