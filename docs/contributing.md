# Contributing

Bug reports, focused feature proposals, documentation improvements and pull requests are
welcome. The repository's
[contribution guide](https://github.com/NewMrPotato/ml-flex-predict/blob/main/CONTRIBUTING.md)
describes editable installation, branch policy, test commands and pull-request expectations.

Documentation changes can be previewed locally with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build. The GitHub workflow treats warnings as
errors and publishes the HTML site from `main`.

Maintainers should use the [release checklist](release-checklist.md) for TestPyPI and PyPI
publication.

```{toctree}
:hidden:

release-checklist
```
