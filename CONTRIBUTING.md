# Contributing to FlexPredict

Thank you for helping improve FlexPredict. Bug fixes, regression tests, documentation,
framework integrations and focused feature proposals are welcome. FlexPredict is currently
an alpha project, so explain the user-facing contract of a change clearly and call out any
breaking behavior.

## Before starting

- Search [existing issues](https://github.com/NewMrPotato/ml-flex-predict/issues) before
  opening a duplicate.
- Open an issue before a large API change or a new framework integration so the design can
  be discussed before significant implementation work.
- Do not include private model artifacts, credentials, customer data or generated build
  distributions in an issue or pull request.

Small bug fixes and documentation corrections can go directly to a pull request.

## Development setup

If you do not have write access, fork the repository on GitHub and clone your fork. Direct
collaborators can clone the upstream repository. Replace `<your-user>` below with your
GitHub username:

```bash
git clone https://github.com/<your-user>/ml-flex-predict.git
cd ml-flex-predict
git remote add upstream https://github.com/NewMrPotato/ml-flex-predict.git
git fetch upstream
git switch dev
git pull --ff-only upstream dev
```

If `upstream` is already the clone's `origin`, use `origin` in the fetch and pull commands
and do not add a second remote.

Create and activate a virtual environment. On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package in editable mode with development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install only the optional integrations needed for the change, for example:

```bash
python -m pip install -e ".[sklearn,pandas]"
python -m pip install -e ".[torch]"
python -m pip install -e ".[tensorflow]"
```

The base installation must continue to require only NumPy. Keep optional framework imports
lazy so `import flexpredict` works without pandas, scikit-learn, PyTorch or TensorFlow.

## Branch and commit workflow

Create a focused branch from `dev`:

```bash
git switch -c fix/descriptive-name
```

Use a short prefix such as `fix/`, `feature/`, `docs/` or `test/`. Keep commits scoped and
write imperative commit messages that describe the result, for example:

```text
Preserve categorical DataFrame columns
```

Pull requests target `dev`, not `main`. The `main` branch contains reviewed release-ready
work and is updated from `dev` through a dedicated pull request.

## Required checks

Run the core quality gate before opening a pull request:

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

Changes to an optional engine or loader must also run the corresponding integration test:

```bash
python -m pytest tests/integration/test_sklearn.py -q
python -m pytest tests/integration/test_torch.py -q
python -m pytest tests/integration/test_tensorflow.py -q
```

Only run integration tests for frameworks installed in the active environment. GitHub
Actions runs the complete cross-platform and optional-framework matrix for every pull
request.

## Tests and compatibility

- Add a regression test for every bug fix.
- Preserve the zero-configuration array path unless a breaking change is intentional and
  documented.
- Test named inputs when feature selection, ordering or schema validation is affected.
- Keep ensemble behavior deterministic and validate class metadata before aggregating
  classification probabilities.
- Use pytest's `tmp_path` for temporary model files and weights. Tests should create their
  own artifacts instead of committing generated binaries.
- Avoid network access in the test suite.
- Keep assertions focused on public behavior rather than private implementation details.

Compatibility fixtures containing model binaries require a clear reason, a documented
generation process and maintainer approval.

## API and implementation expectations

- Preserve `PredictionResult.values` as a non-empty two-dimensional NumPy array.
- Reject unknown configuration instead of silently ignoring it.
- Wrap framework failures in FlexPredict exceptions while preserving the original cause.
- Validate model output before it reaches an ensemble or application boundary.
- Keep framework-specific settings inside `engine_options`.
- Add type annotations to new public APIs and update public exports deliberately.
- Prefer small composable components over framework-specific branches in `Predictor`.

## Documentation and changelog

Update README examples or related documentation whenever public behavior changes. Examples
must be runnable and must not depend on files that are absent from the repository.

Add user-visible changes under `Unreleased` in `CHANGELOG.md`. Breaking changes during the
0.x series must be stated explicitly. Do not change the package version as part of a normal
feature pull request; maintainers do that while preparing a release.

## Pull request checklist

Before requesting review, confirm that:

- the pull request targets `dev`;
- the description explains the problem, solution and user-visible behavior;
- tests cover the change and all relevant local checks pass;
- documentation and `CHANGELOG.md` are updated when required;
- no credentials, model artifacts, `dist/` files or environment-specific files are added;
- unrelated formatting or refactoring is excluded.

## Releases and publishing

Only maintainers publish FlexPredict. Releases use the protected GitHub environments
`testpypi` and `pypi` with OIDC Trusted Publishing; the repository does not store PyPI API
tokens.

Do not run publishing workflows from a contribution branch or upload distributions with
local credentials. Maintainers follow [the release checklist](docs/release-checklist.md),
publish to TestPyPI first, verify a clean installation, and then explicitly approve the
production PyPI deployment.
