from __future__ import annotations

import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from granite_tools.app_types import FingerType
from granite_tools.bigram_scores.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.config import read_config
from granite_tools.easy_rolling import get_easy_rolling_type_mapping
from granite_tools.hands import get_hands_data
from granite_tools.trigram_model.optimizer import get_limit_funcs
from granite_tools.trigram_model.params import TrigramModelParameters
from granite_tools.trigram_model.scorer import (
    get_trigram_score,
    get_trigram_scores,
    max_abs_error,
)
from granite_tools.trigram_types import get_trigram_type

if typing.TYPE_CHECKING:
    KeySeq = tuple[int, ...]
    from granite_tools.config import Config


def get_trigram_data_from_files(
    config_file: str | Path,
    bigram_ranking_file: str | Path,
    trigram_relative_scoring_file: str | Path,
    raw_anchor_ngram_scores_file: str | Path,
) -> pd.DataFrame:
    config = read_config(config_file)
    bigram_scores = load_bigram_and_unigram_scores(
        bigram_ranking_file, raw_anchor_ngram_scores_file
    )
    # TODO: fix this
    trigram_relative_scores = load_trigram_relative_scores(
        trigram_relative_scoring_file
    )
    return get_trigram_data(config, bigram_scores, trigram_relative_scores)


def get_trigram_data(
    config: Config,
    bigram_scores: dict[KeySeq, float],
    trigram_relative_scores: dict[str, dict[str, float]],
    params: TrigramModelParameters | None = None,
) -> pd.DataFrame:
    """Creates trigram score dataframe based on the given configuration and scores.

    The trigram_relative_scores is used for calculating the TARGET trigram score ratios
    ("score_ratio_actual"). The bigram_scores is used for calculating the ESTIMATED
    trigram score ratios ("score_ratio_pred") and estimated trigram scores
    ("estimated_score").
    """
    hands = get_hands_data(config)
    scoresets = TrigramScoreSets.from_relative_trigram_scores(
        trigram_relative_scores, hands
    )
    params = params or TrigramModelParameters.from_config(config)
    mapping = get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
    trigram_scores_iter = get_trigram_scores(
        params, scoresets, hands, bigram_scores, mapping=mapping
    )
    groups = group_trigram_scores(trigram_scores_iter, group_sort_by=max_abs_error)

    get_lower_limit, get_upper_limit = get_limit_funcs(config.limit_multipliers)
    data: dict[str, list[object]] = defaultdict(list)
    for _, scoredicts in groups.items():
        for d_ in reversed(scoredicts):
            d = typing.cast(dict, d_)
            fingers, _ = hands.get_fingers(d["trigram"])
            has_thumb = FingerType.T in fingers
            data["has_thumb"].append(has_thumb)
            meta = d.pop("trigram_score_details")
            data["trigram_type"].append(meta["trigramtype"])
            data["ngram1"].append(meta["ngram1_score"])
            data["ngram2"].append(meta["ngram2_score"])
            data["ngram_sum"].append(meta["ngram1_score"] + meta["ngram2_score"])
            data["base_score"].append(meta["base_score"])
            data["vert2u"].append(meta.get("vert2u", np.nan))
            data["redir"].append(meta.get("redir", np.nan))

            score_ratio_resid = d["score_ratio_pred"] - d["score_ratio_actual"]
            data["score_ratio_resid"].append(score_ratio_resid)
            for postfix in ("_pred", "_actual"):
                r = d[f"score_ratio{postfix}"]
                ll = get_lower_limit([r])[0]
                ul = get_upper_limit([r])[0]
                resid_ul = ul - r
                resid_ll = ll - r
                if score_ratio_resid < 0:
                    score_ratio_scaled_resid = -score_ratio_resid / resid_ll
                else:
                    score_ratio_scaled_resid = score_ratio_resid / resid_ul

                data[f"score_ratio_scaled_resid{postfix}"].append(
                    score_ratio_scaled_resid
                )

            for k, val in d.items():
                data[k].append(val)
    df = pd.DataFrame(data)
    df["score_ratio_relative_resid"] = (
        df["score_ratio_resid"] / df["score_ratio_actual"]
    )

    def get_reference_trigram_data(row):
        d = get_trigram_score(row["reference_trigram"], hands, params, bigram_scores)
        trigram_type = get_trigram_type(row["reference_trigram"], hands, mapping)
        return (
            row["reference_trigram"],
            d["score"],
            trigram_type,
            d.get("vert2u", np.nan),
        )

    def get_ref_bigram_sum(row):
        ref_trigram = row["reference_trigram"]
        indices, _ = hands.where(ref_trigram)
        bigram1 = bigram_scores[(indices[0], indices[1])]
        bigram2 = bigram_scores[(indices[1], indices[2])]
        return bigram1 + bigram2

    def get_min_ngram(row):
        return min(row["ngram1"], row["ngram2"])

    df[["ref_trigram", "ref_trigram_score", "ref_trigram_type", "ref_vert2u"]] = (
        df.apply(get_reference_trigram_data, axis=1, result_type="expand")
    )
    df["ref_bigram_sum"] = df.apply(get_ref_bigram_sum, axis=1)
    df["min_ngram"] = df.apply(get_min_ngram, axis=1)
    df.set_index("trigram", inplace=True)
    return df
