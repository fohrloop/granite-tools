
# Scoring Key Efforts

The [granite-tools](https://github.com/fohrloop/granite-tools) provides a toolkit for creating effort estimates for 1- and 2-key sequences for keyboard layout optimization.

# How to use this?

The full process is:

1. **Config file**: Create a keyboard configuration yaml file. Copy the `examples/keyseq_effort.yml` and use it as a base.
2. **Initial order**: Create initial order with `granite-scorer-baseline`. Let's say you save the result it as `myfile`.
3. **View the initial order** (optional): Use the `granite-scorer-view` to see the initial order. You may also do some fine tuning to the order. Tip: It's possible to use the `granite-scorer-view` also with partial initial order file (e.g. if you find something that's a bit off while working with `granite-scorer-baseline`) 
4. **Create comparison file**. Use the `granite-scorer-compare` with the initial order to create comparisons of different key sequences. 

> [!NOTE]
> If you're using [uv](https://docs.astral.sh/uv/), you will need to run `uv run command` instead of `command`.

## granite-scorer-baseline

Application for creating an initial ordered ngram table. Launch:

```
❯ granite-scorer-baseline <ngram-order-file> <config-file-yml>
```

for example:

```
❯ granite-scorer-baseline myfile examples/keyseq_effort.yml
```

## granite-scorer-view

Application for viewing an ordered ngram table. Launch:

```
❯ granite-scorer-view <ngram-order-file> <config-file-yml>
```

for example:

```
❯ granite-scorer-view myfile examples/keyseq_effort.yml
```


## granite-scorer-compare

Application for viewing an ordered ngram table. Launch:

```
❯ granite-scorer-compare <ngram-order-file|saved-pickle-file> <config-file-yml>
```

for example:

```
❯ granite-scorer-compare myfile examples/keyseq_effort.yml
```

### compare app

- This app uses the [.compare.pickle format](compare-pickle-format.md) for saving the output data.