from __future__ import annotations

import typing
from collections import Counter
from pathlib import Path

if typing.TYPE_CHECKING:
    from granite_tools.app_types import KeySeq


class DuplicateValuesError(RuntimeError): ...


def load_bigram_rankings(file: str | Path) -> list[KeySeq]:
    with open(file, "r") as f:
        lines = f.readlines()

    key_seqs: list[KeySeq] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        key_seq = tuple(map(int, line.split(",")))
        key_seqs.append(key_seq)

    if not key_seqs:
        return []

    most_common_ngram, repeats = Counter(key_seqs).most_common(1)[0]
    if repeats > 1:
        raise DuplicateValuesError(
            f'Duplicate values for "{most_common_ngram}" in "{file}".'
        )
    return key_seqs
