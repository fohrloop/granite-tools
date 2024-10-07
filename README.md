# granite-tools

Character-based ngram analysis tool for keyboard layout optimization. Uses the same ngram format as input as the [Keyboard Layout Optimizer](https://github.com/dariogoetz/keyboard_layout_optimizer) by Dario Götz.


# Commands

## `ngram_show`

Show ngram files. Example:

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

Full help:

```
❯ ngram_show --help
                                                                                                                                                     
 Usage: ngram_show [OPTIONS] NGRAM_SRC                                                                                                               
                                                                                                                                                     
 Show ngrams from a folder or a file.                                                                                                                
                                                                                                                                                     
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    ngram_src      PATH  Path to a folder of ngram files or to a single *-gram.txt file. [required]                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --ngrams-count       -n      INTEGER                     The number of ngrams to show (most common first). [default: 40]                          │
│ --ngram-size         -s      [1|2|3|all]                 Which ngram size to show. [default: all]                                                 │
│ --ignore-case        -i                                  Ignore case when comparing ngrams (i.e. consider 'ab', 'aB', 'Ab', and 'AB' to be the    │
│                                                          same).                                                                                   │
│ --ignore-whitespace  -w                                  Ignore all ngrams which contain whitespace (i.e. Drop ngrams with whitespace).           │
│ --resolution                 INTEGER                     The resolution for printed numbers. Example with resolution of 3: 0.234. This only       │
│                                                          affects the tabular form (not plots).                                                    │
│                                                          [default: 2]                                                                             │
│ --type                       [absolute|cumulative|both]  Type of frequency (ngram score) to show. [default: absolute]                             │
│ --plot                                                   Draw a barplot instead of showing a table.                                               │
│ --raw                                                    Use raw values of the ngram frequencies/counts, instead of normalizing them.             │
│ --help                                                   Show this message and exit.                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

Full help:

```
❯ ngram_compare --help
                                                                                                                                                     
 Usage: ngram_compare [OPTIONS] NGRAM_SRC_REF NGRAM_SRC_OTHER                                                                                        
                                                                                                                                                     
 Compare ngrams from two folders or files.                                                                                                           
                                                                                                                                                     
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    ngram_src_ref        PATH  Path to a folder of ngram files or to a single *-gram.txt file. This is used as the reference. [required]         │
│ *    ngram_src_other      PATH  Path to a folder of ngram files or to a single *-gram.txt file. This is compared to the reference (=other)        │
│                                 [required]                                                                                                        │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --ngrams-count       -n      INTEGER                     The number of ngrams to show (most common first). If using with --diff option, then this │
│                                                          number of ngrams is taken from both corpora, and the union of the top ngrams is shown.   │
│                                                          [default: 40]                                                                            │
│ --ngram-size         -s      [1|2|3|all]                 Which ngram size to show. [default: all]                                                 │
│ --ignore-case        -i                                  Ignore case when comparing ngrams (i.e. consider 'ab', 'aB', 'Ab', and 'AB' to be the    │
│                                                          same).                                                                                   │
│ --ignore-whitespace  -w                                  Ignore all ngrams which contain whitespace (i.e. Drop ngrams with whitespace).           │
│ --resolution                 INTEGER                     The resolution for printed numbers. Example with resolution of 3: 0.234. This only       │
│                                                          affects the tabular form (not plots).                                                    │
│                                                          [default: 2]                                                                             │
│ --type                       [absolute|cumulative|both]  Type of frequency (ngram score) to show. [default: absolute]                             │
│ --plot                                                   Draw a barplot instead of showing a table.                                               │
│ --raw                                                    Use raw values of the ngram frequencies/counts, instead of normalizing them.             │
│ --diff                                                   Show difference using first ngram source as reference.                                   │
│ --swap               -S                                  Swap "ref" and "other" input arguments (`ngram_src_ref` and `ngram_src_other`).          │
│ --help                                                   Show this message and exit.                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Data requirements for `ngram_compare`

See: [Data requirements for `ngram_show`](#data-requirements-for-ngram_show)