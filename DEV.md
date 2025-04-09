# Developer README

## Running tests

All but slow tests:

```
uv run pytest -m "not slow"
```

All tests (before a PR):

```
uv run pytest
```
