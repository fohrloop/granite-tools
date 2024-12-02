KeySeq = tuple[int, ...]


def keyseqs_to_comparisons(keyseqs: list[KeySeq]) -> list[tuple[KeySeq, KeySeq]]:
    """Create comparisons based on ordered sequences of key sequences.

    Parameters
    ----------
    keyseqs: list[tuple[int, ...]]
        The key sequences, which should be ordered from least effort to most
        effort.

    Returns
    -------
    list[tuple[tuple[int, ...], tuple[int, ...]]]
        A chained list of comparisons between the key sequences, where each
        item is a tuple of two key sequences A and B. (A,B) means that A is
        *higher* effort (higher cost score) than B; A > B.
    """
    if len(keyseqs) < 2:
        raise ValueError("At least two key sequences are required.")

    comparisons = []

    for ks1, ks2 in zip(reversed(keyseqs[:-1]), reversed(keyseqs[1:])):
        comparisons.append((ks2, ks1))
    return comparisons
