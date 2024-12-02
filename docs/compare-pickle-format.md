## Details on the .compare.pickle format

This is plain old python dictionary, which may be loaded with the `pickle` module:

```python
import pickle
with open('somefile.compare.pickle', 'rb') as f:
    state = pickle.load(f)
```

### initial_order

- Has the initial order of key sequences
- The numbers correspond to the `key_indices` in the configuration YAML.
- The order is from least to most effort.

### processed_key_sequences & pairs_per_sequence

- The processed key sequences. Each key sequence will be processed once by the compare app (one round).
- There will be `pairs_per_sequence` comparison pairs per each key sequence.

### comparisons_all

- Holds *all* key sequence pair comparisons.
- Each item is a tuple of key sequences. The left one is the one with higher score/cost (=more effort), and the right one is the one with less score/cost/effort.
- First items are from the initial order. If `len(initial_order)` is `I`, there will be `I-1` such pairs.
- After those there will be `pairs_per_sequence` pairs for the *first* key sequence in the `processed_key_sequences`. Then there is `pairs_per_sequence` pairs for the *second* key sequence in the `processed_key_sequences`, and so on, until all processed pairs have `pairs_per_sequence` pairs.