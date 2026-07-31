# Changelog

All notable changes to FlexPredict are documented in this file.

The project follows Semantic Versioning after the 1.0 release. During the 0.x alpha
series, breaking changes are called out explicitly.

## [0.2.0] - Unreleased

### Added

- New `flexpredict` package and `Predictor` API.
- Canonical `PredictionResult` with stable two-dimensional output shapes.
- Schema-aware dictionaries, records, arrays, structured arrays and optional pandas input.
- Callable and transformer preprocessing plus the built-in `Standardizer`.
- Lazy generic, PyTorch and TensorFlow inference engines with a custom engine registry.
- Regression, probability and voting ensembles with strict compatibility checks.
- Global aggregation weights from lists, NumPy arrays and safe `.npy` files.
- Lazy joblib, Keras, complete PyTorch and PyTorch state-dict loaders.
- Cross-platform CI, static typing, linting, package builds and coverage enforcement.
- Metadata inference through the modern sklearn Tags API with a legacy fallback.

### Changed

- Distribution renamed from `unipredict` to `flexpredict`.
- Packaging now uses `pyproject.toml` as its only source of metadata.
- Framework-specific settings are grouped in `engine_options`.

### Removed

- The old `UniPredictor`, `ModelConfig`, `mean/std` constructor parameters and legacy
  output squeezing behavior.
- Mandatory imports of PyTorch and TensorFlow.
