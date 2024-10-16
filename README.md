# granite-tools

Character-based ngram analysis tool for keyboard layout optimization. Uses the same ngram format as input as the [Keyboard Layout Optimizer](https://github.com/dariogoetz/keyboard_layout_optimizer) by Dario Götz.

# Installing

```
python -m pip install git+https://github.com/fohrloop/granite-tools.git
```

# Commands

## `ngram_show`
Show the contents of ngram files.

### Example Showing ngram files (barplot)


```
❯ ngram_show ./ngrams/kla-english -s 2 --plot -n 10
────────────────── kla-english ───────────────────
   1: e␣ ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2.88
   2: ␣t ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2.31
   3: th ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1.98
   4: he ▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1.88
   5: s␣ ▇▇▇▇▇▇▇▇▇▇▇▇▇ 1.75
   6: ␣a ▇▇▇▇▇▇▇▇▇▇▇▇▇ 1.73
   7: d␣ ▇▇▇▇▇▇▇▇▇▇▇▇ 1.52
   8: in ▇▇▇▇▇▇▇▇▇▇▇ 1.46
   9: t␣ ▇▇▇▇▇▇▇▇▇▇ 1.32
  10: er ▇▇▇▇▇▇▇▇▇▇ 1.28
```

### Example: Converting ngram files

This example converts existing ngram files (1-grams.txt, 2-grams.txt, 3-grams.txt) into new ones (and does some conversions):
```
❯ ngram_show /ngrams/somecorpus/ -s 1 -n 0 -w -i --resolution=6 --type=plaintext --exclude-chars="€" > ngrams/newcorpus/1-grams.txt
❯ ngram_show /ngrams/somecorpus/ -s 2 -n 0 -w -i --resolution=6 --type=plaintext --exclude-chars="€" > ngrams/newcorpus/2-grams.txt
❯ ngram_show /ngrams/somecorpus/ -s 3 -n 0 -w -i --resolution=6 --type=plaintext --exclude-chars="€" > ngrams/newcorpus/3-grams.txt
```
- `--exclude-chars="€"`: Removes any ngrams with `€` in them.
- `-i`, which is same as `ignore-case`: converts every upper case character to lowercase (merges ngrams like `aB`, `Ab` in to `ab`)
- `-w,`, which is same as `ignore-whitespace`: removes any ngrams with whitespace
- `--type=plaintext` converts into plaintext format (same as the input format, so just <freq> <chars> pairs on each row)
- `-n 0` takes all ngrams from the file instead of just few first.
- `--resolution=6` saves with resolution of 6 digits.

The new files (1-grams.txt, 2-grams.txt, 3-grams.txt) are all normalized; the frequencies sum up to 100.0 (with floating point accuracy).

### Full help
Full help, see:

 ```
❯ ngram_show --help
```

### Data requirements for `ngram_show`

Assumes that the ngram files follow the same format as the [Keyboard Layout Optimizer](https://github.com/dariogoetz/keyboard_layout_optimizer); `<frequency> <chars>` on each line. For example:

```
7.851906052482727  
7.534280931977296 e
5.621867629166546 t
4.477877757676168 r
4.430916637427879 a
```

and that each set of ngrams are called `1-grams.txt`, `2-grams.txt` and `3-grams.txt` within a single folder, like this:

```
📁 ngrams/
├─📁 leipzig/
| ├─📄 1-grams.txt
| ├─📄 2-grams.txt
| └─📄 3-grams.txt
└─📁 tldr17/
  ├─📄 1-grams.txt
  ├─📄 2-grams.txt
  └─📄 3-grams.txt
```

## `ngram_compare`

Compare two ngram corpora. Example:

```
❯ ngram_compare /home/fohrloop/code/granite-english-ngrams/ngrams/english/ /home/fohrloop/code/granite-code-ngrams/ngrams/code/  --plot -s 3 -n 20  -
i --diff -w
─────────────────────english────────────────────── ───────────────────────code───────────────────────
 1: the   ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2.49                    1 (+2826): --- ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 0.61
 2: ing   ▇▇▇▇▇▇▇▇▇▇▇ 1.43                             2 (   +4): ion ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 0.47
 3: and   ▇▇▇▇▇▇▇▇▇▇ 1.26                              3 (   +6): ent ▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 0.42
 4: hat   ▇▇▇▇▇ 0.61                                   4 (  +33): con ▇▇▇▇▇▇▇▇▇▇▇▇▇ 0.41
 5: her   ▇▇▇▇▇ 0.60                                   5 (   +7): tio ▇▇▇▇▇▇▇▇▇▇▇▇▇ 0.39
 6: ion   ▇▇▇▇▇ 0.56                                   6 (   -5): the ▇▇▇▇▇▇▇▇▇▇ 0.31
 7: tha   ▇▇▇▇ 0.55                                    7 (   -5): ing ▇▇▇▇▇▇▇▇▇▇ 0.30
 8: for   ▇▇▇▇ 0.55                                    8 (  +17): ate ▇▇▇▇▇▇▇▇▇▇ 0.30
 9: ent   ▇▇▇▇ 0.53                                    9 ( +275): sel ▇▇▇▇▇▇▇▇▇▇ 0.29
10: thi   ▇▇▇▇ 0.50                                   10 ( +194): ass ▇▇▇▇▇▇▇▇▇ 0.27
11: all   ▇▇▇▇ 0.47                                   11 (  +68): ect ▇▇▇▇▇▇▇▇▇ 0.26
12: tio   ▇▇▇▇ 0.45                                   12 (  +37): ons ▇▇▇▇▇▇▇▇ 0.25
13: ver   ▇▇▇ 0.42                                    13 (  +88): ort ▇▇▇▇▇▇▇▇ 0.25
14: you   ▇▇▇ 0.42                                    14 ( +240): ser ▇▇▇▇▇▇▇▇ 0.25
15: ter   ▇▇▇ 0.40                                    15 ( +456): elf ▇▇▇▇▇▇▇▇ 0.24
16: ere   ▇▇▇ 0.38                                    16 ( +890): def ▇▇▇▇▇▇▇▇ 0.24
17: his   ▇▇▇ 0.38                                    17 (  +64): ame ▇▇▇▇▇▇▇▇ 0.23
18: ith   ▇▇▇ 0.36                                    18 ( +214): por ▇▇▇▇▇▇▇▇ 0.23
19: wit   ▇▇▇ 0.35                                    19 (   -4): ter ▇▇▇▇▇▇▇ 0.23
20: was   ▇▇▇ 0.33                                    20 (  +30): est ▇▇▇▇▇▇▇ 0.22
25: ate   ▇▇▇ 0.32                                    22 (  -14): for ▇▇▇▇▇▇▇ 0.22
37: con   ▇▇ 0.25                                     60 (  -47): ver ▇▇▇▇▇ 0.16
49: ons   ▇▇ 0.22                                     71 (  -68): and ▇▇▇▇ 0.13
50: est   ▇▇ 0.22                                     95 (  -84): all ▇▇▇▇ 0.11
79: ect   ▇ 0.17                                     111 ( -101): thi ▇▇▇ 0.10
81: ame   ▇ 0.16                                     127 ( -110): his ▇▇▇ 0.10
101: ort  ▇ 0.15                                     141 ( -136): her ▇▇▇ 0.09
204: ass  ▇ 0.10                                     145 ( -127): ith ▇▇▇ 0.09
232: por  ▇ 0.09                                     155 ( -136): wit ▇▇▇ 0.09
254: ser  ▇ 0.08                                     189 ( -173): ere ▇▇ 0.07
284: sel  ▇ 0.08                                     505 ( -498): tha ▇ 0.04
471: elf   0.05                                      603 ( -599): hat ▇ 0.03
906: def   0.03                                     1022 (-1008): you ▇ 0.02
2827: ---  0.00                                     3196 (-3176): was  0.01
```

Full help, see:

```
❯ ngram_compare --help
```

### Data requirements for `ngram_compare`

See: [Data requirements for `ngram_show`](#data-requirements-for-ngram_show)