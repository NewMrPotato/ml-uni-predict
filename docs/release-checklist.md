# FlexPredict release checklist

This checklist is for maintainers preparing an alpha release. FlexPredict publishes through
GitHub Actions and OIDC Trusted Publishing; do not use local PyPI credentials.

## 1. Prepare the release on `dev`

1. Confirm `dev` contains all intended changes and no unrelated work.
2. Move completed entries from `Unreleased` into a versioned section in `CHANGELOG.md` and
   add the release date.
3. Set the intended version in `flexpredict.__version__` and verify package metadata reads
   the same value.
4. Update README installation examples and create release notes for the new version.
5. Confirm breaking changes and migration requirements are explicit.

## 2. Run the release gate

Run locally on a clean checkout:

```bash
python -m pytest --cov=flexpredict --cov-report=term-missing --cov-fail-under=85 -q
python -m ruff check flexpredict tests examples
python -m mypy flexpredict
python -m build
python -m twine check dist/*
```

Install the built wheel into a clean environment and verify the reported version, import and
a basic prediction. Confirm GitHub Actions passes the full Python/OS matrix plus sklearn,
PyTorch and TensorFlow integration jobs.

## 3. Merge, tag and create the GitHub release

1. Open a release pull request from `dev` to `main`.
2. Merge only after every required check passes and the PR is conflict-free.
3. Create an annotated `v<version>` tag on the merge commit and push the tag.
4. Verify the tag and `origin/main` resolve to the same commit.
5. Create a GitHub release from the existing tag. Mark alpha, beta and release-candidate
   versions as pre-releases.
6. Fast-forward `dev` to the new `main` after the merge.

Never move or replace a published release tag.

## 4. Verify through TestPyPI

1. Run **Publish to TestPyPI** from `main` with the exact release tag.
2. Review the build job before approving the protected `testpypi` deployment.
3. Confirm TestPyPI contains one wheel and one source distribution with the expected
   metadata and Python requirement.
4. Install the published wheel into a new environment and run a prediction smoke test.
5. Do not rerun the workflow after a successful upload; package indexes do not permit
   replacing files for an existing version.

## 5. Publish to PyPI

1. Obtain explicit maintainer approval after the TestPyPI smoke test succeeds.
2. Confirm the `pypi` environment permits only `main`, requires a reviewer and uses the
   `publish-pypi.yml` Trusted Publisher.
3. Run **Publish to PyPI** from `main` with the same release tag.
4. Review the build job and explicitly approve the protected `pypi` deployment.
5. Verify the public PyPI metadata and both distribution files.
6. Install `flexpredict==<version>` from the main PyPI index in a clean environment and run
   the same import, version and prediction smoke tests.

If a workflow fails, inspect the first run before retrying. If either distribution was
accepted by an index, do not rebuild or reuse that version number; determine whether the
release must be completed, yanked or superseded by a new version.

## Security rules

- Never commit `.pypirc`, API tokens, generated distributions or model artifacts.
- Never grant `id-token: write` to test, build or repository-wide workflow permissions.
- Keep publishing in a dedicated job that only downloads validated artifacts and runs the
  official PyPA publish action.
- Treat changes to publishing workflows and GitHub environments as security-sensitive code.
