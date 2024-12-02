# granite-tools

Tools used in creation of the [Granite](https://github.com/fohrloop/granite-layout) keyboard layout. Main features:

1. Toolkit for scoring cost for key sequences ("ngrams"): [[Docs](./docs/scoring-key-efforts.md)]
2. Ngram frequency analysis tools: `ngram_show` and `ngram_compare`. [[Docs](./docs/ngram-frequency-analysis.md)]

# Installing

## Install option A: uv

You can use [uv](https://docs.astral.sh/uv/) and let it handle python (virtual) environments for you. In this case

- install [uv](https://docs.astral.sh/uv/)
- Clone this repo or just download the contents
- Use `uv run <command>` instead of `<command>` (must be executed in this directory). It's also possible to activate the virtual environment in the created `.venv` directory (after first run), if you want that.
- This might be the right option for you if you are not familiar with python and virtual environments and.  It's a bit more hassle free.


## Install option B: python virtual environments

- Use **python 3.12** in a fresh virtual environment (this is important as package versions are pinned so you may mess up your system if you don't)

```
python -m pip install git+https://github.com/fohrloop/granite-tools.git
```


