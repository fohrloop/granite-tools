from __future__ import annotations

import typing

from granite_tools.app_types import HAND_TYPES
from granite_tools.trigram_features import TrigramFeatures
from granite_tools.trigram_model.params import TrigramModelParameters
from granite_tools.trigram_types import UnTypableTrigramError

if typing.TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Sequence

    from granite_tools.app_types import HandOrKey, KeySeq
    from granite_tools.easy_rolling import EasyRollingTrigramsMap
    from granite_tools.hands import Hands
    from granite_tools.score_ratios import ScoreRatioEntry


def get_trigram_score(
    ngram: str,
    hands: Hands,
    params: TrigramModelParameters,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
) -> dict:
    """Gets a trigram score using the trigram model."""

    try:
        # TODO: refactor and use indices and keytypes (call hands.where only once)
        features = TrigramFeatures.from_string(
            ngram, hands, mapping=mapping, use_visualization_symbols=True
        )
    except UnTypableTrigramError:
        return dict(score=0, untypable=True)
    indices, keytypes = hands.where(ngram, visualization=True)

    if features.main_type == "onehand":
        return get_onehand_score(indices, bigram_scores, features, params)
    elif features.main_type == "balanced":
        return get_balanced_score(indices, keytypes, bigram_scores, params)
    elif features.main_type == "alternating":
        return get_alternating_score(indices, keytypes, bigram_scores, params)
    else:
        raise ValueError("Unknown trigram type")  # should never happen


def _to_balanced_key_sequences(
    indices: Sequence[int],
    keyseqtypes: Sequence[HandOrKey],
) -> tuple[tuple[KeySeq], tuple[HandOrKey]]:
    return tuple(zip(*_iterate_keyseqtypes(indices, keyseqtypes)))  # type:ignore


def _iterate_keyseqtypes(
    indices: Sequence[int],
    keyseqtypes: Sequence[HandOrKey],
) -> Iterator[tuple[KeySeq, HandOrKey]]:
    prev_keyseqtype = None
    keyseqs: list[int] = []
    for keyseqtype, idx in zip(keyseqtypes, indices):
        if keyseqtype != prev_keyseqtype:
            if prev_keyseqtype is not None:
                yield tuple(keyseqs), prev_keyseqtype
            keyseqs = []
        keyseqs.append(idx)
        prev_keyseqtype = keyseqtype

    if prev_keyseqtype is not None:
        yield tuple(keyseqs), prev_keyseqtype


def _to_alternating_key_sequences(
    indices: Sequence[int],
    keyseqtypes: Sequence[HandOrKey],
) -> tuple[tuple[KeySeq, KeySeq], tuple[HandOrKey, HandOrKey]]:
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    if (
        keyseqtypes[0] != keyseqtypes[2]
        or keyseqtypes[0] == keyseqtypes[1]
        or keyseqtypes[0] not in HAND_TYPES
    ):
        raise ValueError("Only *alternating* trigrams are supported")
    keyseqtype1 = keyseqtypes[0]
    keyseqtype2 = keyseqtypes[1]
    keyseqs: tuple[KeySeq, KeySeq] = (indices[0], indices[2]), (indices[1],)
    return (keyseqs), (keyseqtype1, keyseqtype2)


def get_balanced_score(
    indices: Sequence[int],
    keytypes: Sequence[HandOrKey],
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
):
    """Get score for a balanced type of trigram."""
    key_sequences, key_seq_types = _to_balanced_key_sequences(indices, keytypes)

    if len(key_sequences) != 2:
        raise ValueError("Only balanced trigrams are supported")

    try:
        bigram_score, unigram_score = _get_unigram_and_bigram_scores(
            key_sequences, key_seq_types, bigram_scores
        )

    except UntypableError:
        return dict(score=0, untypable=True)

    base_score, score = calculate_balanced_score(bigram_score, unigram_score, params)

    return dict(
        ngram1_score=bigram_score,
        ngram2_score=unigram_score,
        base_score=base_score,
        balanced_coeff=params.balanced_coeff,
        u_weight_balanced=params.u_weight_balanced,
        score=score,
        trigramtype="balanced",
    )


def calculate_balanced_score(
    bigram_score: float, unigram_score: float, params: TrigramModelParameters
) -> tuple[float, float]:
    base_score = bigram_score + params.u_weight_balanced * unigram_score
    score = params.balanced_coeff * base_score
    return base_score, score


class UntypableError(RuntimeError): ...


def get_alternating_score(
    indices: Sequence[int],
    keytypes: Sequence[HandOrKey],
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
) -> dict:
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    key_sequences, key_seq_types = _to_alternating_key_sequences(indices, keytypes)

    try:
        bigram_score, unigram_score = _get_unigram_and_bigram_scores(
            key_sequences, key_seq_types, bigram_scores
        )

    except UntypableError:
        return dict(score=0, untypable=True)

    base_score, score = calculate_alternating_score(bigram_score, unigram_score, params)

    return dict(
        ngram1_score=bigram_score,
        ngram2_score=unigram_score,
        base_score=base_score,
        score=score,
        u_weight_alternating=params.u_weight_alternating,
        alternating_coeff=params.alternating_coeff,
        trigramtype="alternating",
    )


def calculate_alternating_score(
    bigram_score: float, unigram_score: float, params: TrigramModelParameters
) -> tuple[float, float]:
    base_score = bigram_score + params.u_weight_alternating * unigram_score
    score = params.alternating_coeff * base_score
    return base_score, score


def _get_unigram_and_bigram_scores(
    key_sequences: tuple[KeySeq, ...],
    key_seq_types: tuple[HandOrKey, ...],
    bigram_scores: dict[KeySeq, float],
) -> tuple[float, float]:
    scores = []
    bigram_score, unigram_score = None, None
    for keyseq, keytype in zip(key_sequences, key_seq_types):
        if keytype in HAND_TYPES:
            score = bigram_scores[keyseq]
        else:
            raise UntypableError
        scores.append(score)
        if len(keyseq) == 2:
            bigram_score = score
        elif len(keyseq) == 1:
            unigram_score = score

    if bigram_score is None or unigram_score is None:
        # should never happen
        raise ValueError("Bigram and unigram scores must be found")
    return bigram_score, unigram_score


def get_onehand_score(
    indices: Sequence[int],
    bigram_scores: dict[KeySeq, float],
    features: TrigramFeatures,
    params: TrigramModelParameters,
):
    d = get_onehand_base_score(indices, bigram_scores, params)
    multiplier = get_feature_multiplier(features, params)

    d = multiplier | d
    score = multiplier["feature_multiplier"] * d["base_score"]
    return dict(**d, score=score, trigramtype="onehand")


def get_onehand_base_score(
    indices: Sequence[int],
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
):
    bigram_ab_score = bigram_scores[indices[0], indices[1]]
    bigram_bc_score = bigram_scores[indices[1], indices[2]]
    skipgram_ac_score = bigram_scores[indices[0], indices[2]]

    base_score = calculate_onehand_score(
        bigram_ab_score,
        bigram_bc_score,
        skipgram_ac_score,
        params.w_ac_one,
    )

    return dict(
        ngram1_score=bigram_ab_score,
        ngram2_score=bigram_bc_score,
        skipgram_score=skipgram_ac_score,
        base_score=base_score,
    )


def calculate_onehand_score(
    bigram1_score: float, bigram2_score: float, skipgram_score: float, w_ac_one: float
) -> float:
    a_norm = get_normalization_factor(w_ac_one)
    return a_norm * (bigram1_score + bigram2_score + w_ac_one * skipgram_score)


def get_normalization_factor(w_ac_one: float) -> float:
    """Get the normalization factor (a_norm) for onehand trigram base score."""
    return 3 / (4 + 2 * w_ac_one)


def get_feature_multiplier(
    features: TrigramFeatures,
    params: TrigramModelParameters,
):

    if features.vert2u == "v1x":
        vert2u = 1.0 + params.vert2u_coeff
    elif features.vert2u == "v2x":
        vert2u = 1.0 + 2 * params.vert2u_coeff
    elif features.vert2u is None:
        vert2u = 1.0
    else:
        raise ValueError(f"Unknown vert2u value {features.vert2u}")

    if features.single_finger_pattern == "SFB":
        sfp = params.sfb_in_onehand_coeff
    elif features.single_finger_pattern == "SFT":
        sfp = params.sft_coeff
    elif features.single_finger_pattern == "SFTb":
        sfp = params.sftb_coeff
    elif features.single_finger_pattern == "SFS":
        sfp = params.sfs_coeff
    elif features.single_finger_pattern == "SFSb":
        sfp = params.sfsb_coeff
    elif features.single_finger_pattern is None:
        sfp = 1.0
    else:
        raise ValueError(
            f"Unknown single_finger_pattern value {features.single_finger_pattern}"
        )

    if features.redir == "redir":
        redir = params.redir_coeff
    elif features.redir is None:
        redir = 1.0
    else:
        raise ValueError(f"Unknown redir value {features.redir}")

    if features.easy_rolling == "easy-rolling":
        easy_rolling = params.easy_rolling_coeff
    elif features.easy_rolling is None:
        easy_rolling = 1.0
    else:
        raise ValueError(f"Unknown easy_rolling value: {features.easy_rolling}")

    feature_multiplier = easy_rolling * vert2u * sfp * redir
    return dict(
        feature_multiplier=feature_multiplier,
        easy_rolling_multiplier=easy_rolling,
        vert2u_multiplier=vert2u,
        sfp_multiplier=sfp,
        redir_multiplier=redir,
    )


class TrigramScoreDict(typing.TypedDict):
    trigram: str
    reference_trigram: str
    score_ratio_actual: float
    score_ratio_pred: float
    estimated_score: float
    estimated_score_details: dict


def get_trigram_scores(
    model_params: TrigramModelParameters,
    trigram_score_ratios: list[ScoreRatioEntry],
    hands: Hands,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
) -> list[TrigramScoreDict]:
    """Calculate trigram scores using the trigram model.

    Parameters
    ----------
    model_params : TrigramModelParameters
        The model parameters for the trigram model.
    trigram_score_ratios : list[ScoreRatioEntry]
        The list of trigram score ratios.
    bigram_scores:
        Bigram scores.

    Returns
    -------
    list[TrigramScoreDict]
        Trigam score dictionaries with the trigram scores and additional
        information. Meaning of the keys:

        - score_ratio_actual: The target score is the target score from the manually
          given scores (from the trigram scoring file). Because it's scaled score it
          means that it's scaled by dividing by the score of the reference trigram.
        - score_ratio_pred: The estimated score is the score estimated by the
          model. This is also scaled by dividing by the score of the reference trigram.
          In other words, it is model_score(trigram)/model_score(reference_trigram).
    """

    def get_trigram_score_(trigram: str) -> dict:
        """Convenience function which handles expection(s) and makes it easier to call
        get_trigram_score."""
        try:
            scoredct = get_trigram_score(
                trigram, hands, model_params, bigram_scores, mapping=mapping
            )
        except KeyError as err:
            raise RuntimeError(
                f"Trigram {trigram} could not be scored! The scores for indices {str(err)} not found in bigram scores! (see the Traceback for more details)"
            ) from err
        if scoredct.get("untypable"):
            raise UntypableError(
                f'The trigram "{trigram}" is untypable with the given configuration file! Check that the `symbols_visualization` in the configuration contains all the required symbols!'
            )
        return scoredct

    scores: list[TrigramScoreDict] = []
    for entry in trigram_score_ratios:
        trigram = entry["ngram"]
        ref_trigram = entry["ref"]
        ref_score_dct = get_trigram_score_(ref_trigram)
        ref_score = ref_score_dct["score"]

        estimated_score_dct = get_trigram_score_(trigram)
        estimated_score = estimated_score_dct["score"]
        score_ratio_pred = estimated_score / ref_score
        scores.append(
            dict(
                trigram=trigram,
                reference_trigram=ref_trigram,
                score_ratio_actual=entry["score_ratio"],
                score_ratio_pred=score_ratio_pred,
                estimated_score=estimated_score,
                estimated_score_details=estimated_score_dct,
            )
        )

    return scores


def max_abs_error(dcts: Sequence[TrigramScoreDict]) -> float:
    return max([abs(_get_error(d)) for d in dcts])


def average_abs_error(dcts: Sequence[TrigramScoreDict]) -> float:
    return sum([abs(_get_error(d)) for d in dcts]) / len(dcts)


def _get_error(d: TrigramScoreDict) -> float:
    """Calculates a loss values for a single trigram score pair (trigram model scores)"""
    return d["score_ratio_pred"] - d["score_ratio_actual"]
