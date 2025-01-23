from __future__ import annotations

import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from granite_tools.app_types import FingerType
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreSets,
    get_score,
    group_trigram_scores,
    iter_trigrams_scores,
    load_trigram_scores,
    max_abs_error,
)
from granite_tools.scorer.trigramtype import (
    get_easy_rolling_type_mapping,
    get_trigram_type,
)

if typing.TYPE_CHECKING:
    KeySeq = tuple[int, ...]
    from granite_tools.config import Config


def get_trigram_data_from_files(
    config_file: str | Path,
    bigram_ranking_file: str | Path,
    trigram_scoring_file: str | Path,
) -> pd.DataFrame:
    config = read_config(config_file)
    bigram_scores = load_bigram_and_unigram_scores(bigram_ranking_file)
    trigram_scores = load_trigram_scores(trigram_scoring_file)
    return get_trigram_data(config, bigram_scores, trigram_scores)


def get_trigram_data(
    config: Config,
    bigram_scores: dict[KeySeq, float],
    trigram_scores: dict[str, dict[str, float]],
    params: TrigramModelParameters | None = None,
) -> pd.DataFrame:

    hands = get_hands_data(config)
    scoresets = TrigramScoreSets.from_trigram_scores(trigram_scores, hands)
    params = params or TrigramModelParameters.from_config(config)
    mapping = get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
    trigram_scores_iter = iter_trigrams_scores(
        params, scoresets, hands, bigram_scores, mapping=mapping
    )
    groups = group_trigram_scores(trigram_scores_iter, group_sort_by=max_abs_error)

    data: dict[str, list[object]] = defaultdict(list)
    for _, scoredicts in groups.items():
        for d in reversed(scoredicts):

            fingers, _ = hands.get_fingers(d["trigram"])
            has_thumb = FingerType.T in fingers
            data["has_thumb"].append(has_thumb)
            meta = d.pop("estimated_score_details")  # type: ignore
            data["trigram_type"].append(meta["trigramtype"])
            data["ngram1"].append(meta["ngram1_score"])
            data["ngram2"].append(meta["ngram2_score"])
            data["ngram_sum"].append(meta["ngram1_score"] + meta["ngram2_score"])
            data["base_score"].append(meta["base_score"])
            data["vert2u"].append(meta.get("vert2u", np.nan))
            for key, val in d.items():
                data[key].append(val)
    df = pd.DataFrame(data)
    df["scaled_err"] = df["scaled_estimated_score"] - df["scaled_target_score"]
    df["relative_err"] = df["scaled_err"] / df["scaled_target_score"]

    def get_reference_trigram_data(row):
        d = get_score(row["reference_trigram"], hands, params, bigram_scores)
        trigram_type = get_trigram_type(row["reference_trigram"], hands, mapping)
        return d["score"], trigram_type, d.get("vert2u", np.nan)

    def get_ref_bigram_sum(row):
        ref_trigram = row["reference_trigram"]
        indices, _ = hands.where(ref_trigram)
        bigram1 = bigram_scores[(indices[0], indices[1])]
        bigram2 = bigram_scores[(indices[1], indices[2])]
        return bigram1 + bigram2

    def get_min_ngram(row):
        return min(row["ngram1"], row["ngram2"])

    df[["ref_trigram_score", "ref_trigram_type", "ref_vert2u"]] = df.apply(
        get_reference_trigram_data, axis=1, result_type="expand"
    )
    df["ref_bigram_sum"] = df.apply(get_ref_bigram_sum, axis=1)
    df["min_ngram"] = df.apply(get_min_ngram, axis=1)
    df.set_index("trigram", inplace=True)
    return df
