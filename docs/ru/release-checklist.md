# Чек-лист релиза FlexPredict

Чек-лист для maintainer'а, выпускающего alpha-релиз. Публикация выполняется через GitHub
Actions и OIDC Trusted Publishing; локальные учётные данные PyPI не используются.

## 1. Подготовка в `dev`

1. Убедиться, что `dev` содержит только изменения релиза.
2. Перенести готовые пункты `Unreleased` в версионный раздел `CHANGELOG.md` и указать дату.
3. Установить версию в `flexpredict.__version__` и сверить package metadata.
4. Обновить примеры установки в README и создать release notes.
5. Явно описать breaking changes и миграцию.

## 2. Проверки перед релизом

На чистом checkout выполнить:

```bash
python -m pytest --cov=flexpredict --cov-report=term-missing --cov-fail-under=85 -q
python -m ruff check flexpredict tests examples
python -m mypy flexpredict
python -m build
python -m twine check dist/*
```

Установить wheel в чистое окружение и проверить версию, импорт и базовый прогноз. Убедиться,
что GitHub Actions прошёл матрицу Python/OS и интеграции sklearn, PyTorch и TensorFlow.

## 3. Merge, tag и GitHub release

1. Открыть release PR из `dev` в `main`.
2. Выполнить merge только после всех обязательных проверок.
3. Создать annotated tag `v<version>` на merge commit и отправить его.
4. Убедиться, что tag и `origin/main` указывают на один commit.
5. Создать GitHub release из существующего tag. Alpha, beta и RC отметить pre-release.
6. Fast-forward ветку `dev` к новому `main`.

Никогда не перемещать и не заменять опубликованный release tag.

## 4. Проверка через TestPyPI

1. Запустить **Publish to TestPyPI** из `main` с точным release tag.
2. Проверить build job перед подтверждением защищённого deployment.
3. Проверить wheel, source distribution, metadata и Python requirement в TestPyPI.
4. Установить опубликованный wheel в новое окружение и выполнить smoke test.
5. После успешной загрузки не перезапускать workflow: индекс не заменяет файлы версии.

## 5. Публикация в PyPI

1. Получить явное одобрение maintainer'а после успешного TestPyPI smoke test.
2. Проверить ограничения environment `pypi`, reviewer и Trusted Publisher.
3. Запустить **Publish to PyPI** из `main` с тем же tag.
4. Проверить build job и явно одобрить защищённый deployment.
5. Проверить metadata и оба distribution-файла в публичном PyPI.
6. Установить `flexpredict==<version>` из PyPI в чистое окружение и повторить smoke tests.

Если workflow завершился ошибкой, сначала исследовать первый запуск. Если индекс принял хотя
бы один distribution, не пересобирать и не использовать этот номер версии повторно — решить,
нужно ли завершить релиз, отозвать его или выпустить следующую версию.

## Правила безопасности

- Не коммитить `.pypirc`, API tokens, distributions и модельные артефакты.
- Не выдавать `id-token: write` тестовым, build или общим workflow permissions.
- Публиковать в отдельном job, который скачивает проверенные артефакты и запускает официальный
  PyPA action.
- Считать изменения publishing workflows и GitHub environments security-sensitive кодом.
