# Участие в разработке

Сообщения об ошибках, небольшие предложения, улучшения документации и pull requests
приветствуются. Установка для разработки, правила веток и обязательные проверки описаны в
[CONTRIBUTING.md](https://github.com/NewMrPotato/ml-flex-predict/blob/main/CONTRIBUTING.md).

Документация проверяется локально так:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Workflow считает предупреждения ошибками и публикует сайт из `main`. Maintainer'ам следует
использовать [чек-лист релиза](release-checklist.md).

```{toctree}
:hidden:

release-checklist
```
