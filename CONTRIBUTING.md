# Contributing to FlexPredict

FlexPredict keeps the base installation small and tests optional frameworks separately.
Create changes from the `dev` branch and keep framework imports lazy so importing
`flexpredict` continues to require only NumPy.

## Local setup

```bash
python -m pip install -e ".[dev]"
```

Install only the optional integrations needed for the change, for example:

```bash
python -m pip install -e ".[sklearn,pandas]"
```

## Required checks

```bash
python -m pytest --cov=flexpredict --cov-report=term-missing --cov-fail-under=85 -q
python -m ruff check flexpredict tests examples
python -m mypy flexpredict
python -m build
python -m twine check dist/*
python -m examples.basic
python -m examples.schema_and_preprocessing
python -m examples.ensemble
```

Changes to an optional engine or loader should also run its integration test file. Tests
must create temporary model artifacts with pytest's `tmp_path`; do not commit generated
weights or model binaries unless a compatibility fixture specifically requires one.

## API expectations

- Keep the zero-configuration array path working.
- Reject unknown configuration instead of silently ignoring it.
- Preserve `PredictionResult.values` as a non-empty two-dimensional array.
- Wrap framework failures in FlexPredict exceptions while preserving the original cause.
- Add changelog entries for user-visible behavior and call out breaking changes during 0.x.
