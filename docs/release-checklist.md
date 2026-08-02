# FlexPredict release checklist

Use this checklist before tagging an alpha release. Publishing is intentionally manual until
the package name, metadata and first release artifact have been verified on PyPI.

1. Confirm `CHANGELOG.md` describes all user-visible changes and replace `Unreleased` with
   the release date.
2. Confirm `flexpredict.__version__` contains the intended version.
3. Run the unit suite, coverage gate, Ruff and mypy on a clean checkout.
4. Run sklearn, PyTorch and TensorFlow integration jobs in GitHub Actions.
5. Build both artifacts with `python -m build` and validate them with
   `python -m twine check dist/*`.
6. Install the wheel into a clean environment and verify `import flexpredict`, a basic
   prediction and the reported version.
7. Merge `dev` into `main`, create an annotated version tag and publish the GitHub release.
8. Upload to TestPyPI first. Verify installation by distribution name (`flexpredict`) before
   publishing the same artifacts to PyPI.

Never rebuild artifacts between TestPyPI and PyPI; upload the exact files that passed the
checks. Never commit `.pypirc`, API tokens, generated distributions or model artifacts.
