"""Remove ngrams from a scoreratios file

WARNING: This script modifies the scoreratios file in place.

Usage:

    python granite_tools/scripts/scoreratios_modify.py CONFIG_FILE SCORERATIO_FILE [--remove NGRAMS]

Example:

    python granite_tools/scripts/scoreratios_modify.py examples/config.yml data/granite.bigram.scoreratios.yml --remove Q SD

"""

import argparse
import typing

import yaml

from granite_tools.bigram_scores.score_ratio_template import save_score_ratios
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove ngrams from a scoreratios file"
    )
    parser.add_argument(
        "config_file", help="The path to the keyboard configuration (YAML) file."
    )
    parser.add_argument("scoreratio_file", help="The path to the scoreratios file.")

    parser.add_argument(
        "--remove",
        nargs="*",
        type=str,
        default=[],
        help="Ngrams to remove. If listed, removes _both_ sides; Q removes also P on a regular QWERTY layout. Example: --remove Q SD",
    )
    args = parser.parse_args()

    with open(args.scoreratio_file) as f:
        d = yaml.safe_load(f)
    config = read_config(args.config_file)
    hands = get_hands_data(config)

    ngrams_to_remove = args.remove
    print(f"Removing ngrams: {ngrams_to_remove}")
    print("Ngram pairs before:", len(d))

    def remove_ngram(d, keyseq: tuple[int, ...]):
        out = []
        ngrams_to_remove = set()
        for hand in ("Left", "Right"):
            ngrams_to_remove.add(
                hands.get_symbols_visualization(
                    typing.cast(typing.Literal["Left", "Right"], hand), keyseq
                )
            )

        for x in d:
            if x["ngram"] in ngrams_to_remove or x["ref"] in ngrams_to_remove:
                continue
            out.append(x)
        return out

    for ngram in ngrams_to_remove:
        keyseq, _ = hands.where(ngram)
        d = remove_ngram(d, keyseq)

    print("Ngram pairs after:", len(d))

    refs = []
    ngrams = []
    score_ratios = []
    for entry in d:
        refs.append(entry["ref"])
        ngrams.append(entry["ngram"])
        score_ratios.append(entry["score_ratio"])
    save_score_ratios(args.scoreratio_file, ngrams, refs, score_ratios=score_ratios)
    print(f"Overwritten '{args.scoreratio_file}' with updated data.")
