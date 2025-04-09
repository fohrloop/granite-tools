from __future__ import annotations

import typing
from pathlib import Path
from typing import TypedDict

import yaml

if typing.TYPE_CHECKING:

    from granite_tools.app_types import KeySeq
    from granite_tools.hands import Hands


class ScoreRatioYamlEntry(TypedDict):
    ref: str
    ngram: str
    score_ratio: float


class ScoreRatioEntry(TypedDict):
    ref: str
    ngram: str
    score_ratio: float
    ref_keyseq: KeySeq
    ngram_keyseq: KeySeq


def load_score_ratio_entries(
    scoreratio_file: str | Path,
    hands: Hands,
) -> list[ScoreRatioEntry]:
    """
    Load the score ratio entries from a YAML file.

    Parameters
    ----------
    scoreratio_file : str | Path
        The path to the YAML file containing the score ratio entries.
    hands : Hands
        The hands object to use for the trigram type checking. Created with the
        get_hands_data() function.

    Returns
    -------
    list[ScoreRatioEntry]
        A list of score ratio entries.

    Notes
    -----
    The expected format of the YAML file is as follows:
    - ref: "ABC"
      ngram: "DEF"
      score_ratio: 1.5
    - ref: "EFG"
      ngram: "HIJ"
      score_ratio: 2.2

    Each score ratio is the quotient score(ngram)/score(ref_ngram). The symbols in the
    trigrams should be included in the `symbols_visualization` setting in the granite
    configuration file used to create the hands object.
    """
    with open(scoreratio_file) as f:
        raw_entries = typing.cast(list[ScoreRatioYamlEntry], yaml.safe_load(f))
    return create_score_ratio_entries(raw_entries, hands)


def create_score_ratio_entries(
    score_ratio_yaml_list: list[ScoreRatioYamlEntry],
    hands: Hands,
) -> list[ScoreRatioEntry]:
    """
    Create score ratio entries based on a dictionary

    """
    out = []
    for item in score_ratio_yaml_list:
        ref_keyseq, _ = hands.where(item["ref"], visualization=True)
        ngram_keyseq, _ = hands.where(item["ngram"], visualization=True)

        out.append(
            ScoreRatioEntry(
                ref=item["ref"],
                ngram=item["ngram"],
                score_ratio=item["score_ratio"],
                ref_keyseq=ref_keyseq,
                ngram_keyseq=ngram_keyseq,
            )
        )

    return out
