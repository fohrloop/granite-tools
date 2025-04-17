"""This module provides functions for creating bigram score ratio template files"""

from __future__ import annotations

import math
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from typing import Sequence, TypeVar

    T = TypeVar("T")


def get_group_sizes(total_items: int, group_max_size: int) -> list[int]:
    """Get the sizes of groups when dividing `total_items` into groups of `group_max_size`
    or smaller. The groups are divided in a balanced way, i.e. the sizes of the groups
    are as equal as possible (maximum difference of one).

    If some groups are smaller than others, they are placed at the beginning of the
    list"""
    groups = math.ceil(total_items / group_max_size)
    parts, remainder = divmod(total_items, groups)
    return [parts] * (groups - remainder) + [parts + 1] * remainder


def select_every_nth_item(items: Sequence[T], n: int) -> Sequence[T]:
    """Selects every "nth" item from the list of items, if possible. If not possible
    directly, selects "as closely as possible", by first selecting every "n-1"th, and
    then every "nth". The first and last items are always included."""
    group_sizes = get_group_sizes(len(items) - 1, n)
    out = [items[0]]
    cur = 0
    for group_size in group_sizes:
        cur += group_size
        out.append(items[cur])
    return out


score_ratio_entry_template = """- ref: "{ref}"
  ngram: "{ngram}"
  score_ratio: {score_ratio}

"""


def save_score_ratios(
    outfile: str | Path,
    ngrams: Sequence[str],
    ref_ngrams: Sequence[str],
    score_ratios: Sequence[float] | None = None,
) -> None:
    """Save the score ratios to a YAML file. The ngrams and ref_ngrams are lists of
    ngrams and their references. The score_ratios is a list of score ratios, which is
    optional. If not provided, the score ratio is set to 1.0 for all ngram pairs."""
    with open(outfile, "w") as f:
        for i, (ngram, ref) in enumerate(zip(ngrams, ref_ngrams)):
            score_ratio = score_ratios[i] if score_ratios is not None else 1

            txt = score_ratio_entry_template.format(
                ngram=ngram, ref=ref, score_ratio=score_ratio
            )
            f.write(txt)
