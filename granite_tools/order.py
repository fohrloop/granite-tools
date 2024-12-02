from collections import Counter
from pathlib import Path


class DuplicateValuesError(RuntimeError): ...


def load_key_seqs_from_file(file: str | Path) -> list[tuple[int, ...]]:
    with open(file, "r") as f:
        lines = f.readlines()

    key_seqs = []

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
