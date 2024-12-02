import logging

import choix  # type: ignore

KeySeq = tuple[int, ...]


def get_scores(
    comparison_data: list[tuple[KeySeq, KeySeq]],
) -> dict[KeySeq, float]:

    unique_key_sequences = {ks for pair in comparison_data for ks in pair}
    mapping = {i: ks for i, ks in enumerate(unique_key_sequences)}
    reverse_mapping = {ks: i for i, ks in mapping.items()}
    data = []
    for a, b in comparison_data:
        data.append((reverse_mapping[a], reverse_mapping[b]))

    alpha = 1e-25
    while True:
        try:
            scores_arr = choix.opt_pairwise(
                n_items=len(mapping),
                data=data,
                alpha=alpha,
                max_iter=100,
            )
            break
        except ValueError:
            logging.warning("Failed with alpha = %s", alpha)
            alpha *= 10

    scores = {mapping[i]: float(score) for i, score in enumerate(scores_arr)}
    return scores
