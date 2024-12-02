
# Scoring Key Efforts

The [granite-tools](https://github.com/fohrloop/granite-tools) provides a toolkit for creating effort estimates for 1- and 2-key sequences for keyboard layout optimization.

# How to use this?

The full process is:

1. **Config file**: Create a keyboard configuration yaml file. Copy the `examples/keyseq_effort.yml` and use it as a base.
2. **Initial order**: Create initial order with `sort_app`. Let's say you save the result it as `myfile`.
3. **View the initial order** (optional): Use the `viewer` to see the initial order. You may also do some fine tuning to the order. Tip: It's possible to use the `viewer` also with partial initial order file (e.g. if you find something that's a bit off while working with `sort_app`) 
4. **Create comparison file**. Use the `compare_app` with the initial order to create comparisons of different key sequences. 


## sort_app

Application for creating an ordered ngram table. Launch:

```
❯ uv run textual run app/sort_app/sort_app.py <ngram-order-file> <config-file-yml>
```

for example:

```
❯ uv run textual run app/sort_app/sort_app.py  somefile examples/keyseq_effort.yml
```

## viewer_app

Application for viewing an ordered ngram table. Launch:

```
❯ uv run textual run app/viewer/viewer_app.py <ngram-order-file> <config-file-yml>
```

for example:

```
❯ uv run textual run app/viewer/viewer_app.py somefile examples/keyseq_effort.yml
```

## compare app

- This app uses the [.compare.pickle format](compare-pickle-format.md) for saving the output data.