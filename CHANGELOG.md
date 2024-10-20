# Changelog

## granite-tools 0.3.0
🗓️ 2024-10-20

- ✨ Add parameter `--include-chars` which can be used to exclude any ngrams with the given characters (for `ngram_show`)


## granite-tools 0.2.0
🗓️ 2024-10-16

### ngram_show & ngram_compare
- ✨ Show all the ngrams if `-n, --ngrams-count` of the is set to 0.
- ✨ Add parameter `--exclude-chars` which can be used to exclude any ngrams with the given characters.
- 🚨 Rename `--type` to `--freq` (frequency type)
- 🚨 Remove `--plot` and replace with `--type` which may be 'table', 'plot' or 'plaintext' (new). The plaintext format is supported by dariogoetz/keyboard_layout_optimizer.
- 🚨 Now sort results also by characters in ngram (after the frequency. In practice should not change anything as the freq values are floats and practically never equal. But helps in testing)

## granite-tools 0.1.1
🗓️ 2024-10-08

- 🐞 Fix crash with ngram_compare when reference dataset contains ngrams which are not present in the other ngram dataset. Example of new printout: `??? (???): И  0.00`.

## granite-tools 0.1.0
🗓️ 2024-10-07

- The initial version