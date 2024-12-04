from __future__ import annotations

import sys
import tomllib
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize

from granite_tools.app_types import (
    ONEHAND_ROLLING_TYPES,
    ONEHAND_TRIGRAM_TYPES,
    FingerType,
    OnehandRollingType,
    OnehandTrigramType,
)
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.trigramtype import (
    get_easy_rolling_type_mapping,
    get_trigram_type,
)
from granite_tools.utils import get_linear_scaling_function

if typing.TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Iterable, Sequence

    from granite_tools.app_types import HandOrKey, KeySeq
    from granite_tools.config import Config
    from granite_tools.hands import Hand, Hands
    from granite_tools.scorer.trigramtype import EasyRollingTrigramsMap

DEFAULT_BALANCED_B_COEFF = 1.0
DEFAULT_UNIGRAM_COEFF = 1.5
DEFAULT_SKIPGRAM_B_COEFF = 1.0
DEFAULT_SFT_COEFF = 2.0
DEFAULT_SFB_IN_ONEHAND_COEFF = 1.03
DEFAULT_VERT2U_COEFF = 1.40
DEFAULT_DIRCHANGE_COEFF = 1.1
DEFAULT_EASY_ROLLING_COEFF = 1.0
DEFAULT_BIGRAM_RAW_RANGE_MAX = 5.0
DEFAULT_BIGRAM_SCALING_EXPONENT = 1.0


HAND_TYPES = ("Left", "Right")


def get_vert2u_multiplier(indices: Sequence[int], hand: Hand, coeff: float) -> float:
    """Get the vertical 2 unit ping pong multiplier. The condition when the multiplier
    if given is when there is (2u, -2u) or (-2u, 2u) vertical movement in the trigram
    and the 2nd key is not typed with the middle finger, but either with pinky, ring or
    index. In all other cases, the multiplier is 1.0."""

    matrix_positions = [hand.matrix_positions[idx] for idx in indices]
    if len(matrix_positions) != 3:
        raise ValueError("Only trigrams are supported")

    first_row, mid_row, last_row = [item[1] for item in matrix_positions]

    movements = ((first_row - mid_row), (mid_row - last_row))
    if movements in {(2, -2), (-2, 2)}:
        # 2 * 2 unit jump
        rows = {x[1] for x in hand.matrix_positions.values()}
        top_row = min(rows)  # row numbering grows from top to bottom
        fingers = [hand.get_finger(idx) for idx in indices]
        if FingerType.T in fingers:
            return 1.0
        if fingers[1] == FingerType.M and mid_row == top_row:
            return 1.0
        return coeff
    return 1.0


def get_dirchange_multiplier(indices: Sequence[int], hand: Hand, coeff: float) -> float:
    """Get the direcation change multiplier. This penalizes large direction changes
    (redirs) which have 180 degree (or very close to 180 degree) angle between the
    vectors of the bigrams, and returns coeff. Othewise, returns 1.0

    Parameters
    ----------
    matrix_positions : list[tuple[int, int]]
        The matrix positions of the ngram. The values must be (column, row) positions on
        the keyboard matrix."""

    matrix_positions = [hand.matrix_positions[idx] for idx in indices]
    if len(matrix_positions) != 3:
        raise ValueError("Only trigrams are supported")

    fingers = [hand.get_finger(idx) for idx in indices]
    if FingerType.T in fingers:
        return 1.0
    if fingers[0] == fingers[2]:
        return 1.0

    cos_theta = get_cosine_of_trigram_angle(matrix_positions)
    multiplier = _get_multiplier_from_cosine(cos_theta, coeff)
    return multiplier


def _get_multiplier_from_cosine(cos_theta: float, coeff: float):

    theta = np.rad2deg(np.arccos(cos_theta))

    if np.isnan(theta):
        return 1.0

    if abs(theta - 180) < 0.01:  # around 90 degree angle
        return coeff

    return 1.0


def get_cosine_of_trigram_angle(
    trigram_matrix_positions: list[tuple[int, int]]
) -> float:
    u, v = get_bigram_vectors(trigram_matrix_positions, invert_y=True)
    return get_cos_between_two_vectors(u, v)


def get_bigram_vectors(
    trigram_matrix_positions: list[tuple[int, int]], invert_y: bool = True
):
    """Gets the two bigram vetors from the matrix positions of a trigram

    The invert_y is used to invert the y-axis because in the matrix_positions, the
    positive x-direction is to the right (correct) and the positive y-direction is
    downwards, which we change to upwards to make calculations easier."""
    if len(trigram_matrix_positions) != 3:
        raise ValueError("Only trigrams are supported")
    first, mid, last = trigram_matrix_positions
    u = (mid[0] - first[0], first[1] - mid[1] if invert_y else mid[1] - first[1])
    v = (last[0] - mid[0], mid[1] - last[1] if invert_y else last[1] - mid[1])
    return u, v


def get_cos_between_two_vectors(u, v) -> float:
    product_of_lenghts = norm(u) * norm(v)
    if product_of_lenghts == 0:
        return np.nan
    cos_of_angle = np.dot(u, v) / product_of_lenghts
    return np.clip(cos_of_angle, -1, 1)


def get_angle_between_two_vectors(u, v) -> float:
    # unused?? (could remove?)
    cos_of_angle = get_cos_between_two_vectors(u, v)
    angle_rad = np.arccos(cos_of_angle)
    return float(np.rad2deg(angle_rad))


def get_sft_multiplier(indices, hand: Hand, coeff: float) -> float:
    """Get the same finger trigram multiplier.

    Parameters
    ----------
    indices : list[int]
        The key indices of the ngram.
    hand : Hand
        The hand of the ngram.

    Returns
    -------
    float
        The multiplier for the same finger trigram."""
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    fingers = [hand.get_finger(idx) for idx in indices]
    if fingers[0] is None:
        raise ValueError("Finger not found")
    if len(set(fingers)) == 1:
        return coeff
    return 1.0


def get_sfb_in_onehand_multiplier(indices, hand: Hand, coeff: float) -> float:
    """Get the "same finger bigram in a trigram" multiplier. Note that SFT is not
    considered as SFB!

    Parameters
    ----------
    indices : list[int]
        The key indices of the ngram.
    hand : Hand
        The hand of the ngram.

    Returns
    -------
    float
        The multiplier for the same finger trigram."""
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    fingers = [hand.get_finger(idx) for idx in indices]
    if len(set(fingers)) != 2:
        return 1.0
    if fingers[0] == fingers[1] or fingers[1] == fingers[2]:
        return coeff
    return 1.0


def get_trigram_score(
    ngram: str,
    hands: Hands,
    params: TrigramModelParameters,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
) -> dict:
    indices, keytypes = hands.where(ngram)
    trigramtype = get_trigram_type(ngram, hands, mapping=mapping)

    if trigramtype in ONEHAND_TRIGRAM_TYPES:
        trigramtype = typing.cast(OnehandTrigramType, trigramtype)
        hand = hands.left if keytypes[0] == "Left" else hands.right
        return get_onehand_score(
            indices, hand, bigram_scores, params, trigramtype=trigramtype
        )
    elif trigramtype == "balanced":
        return get_balanced_score(indices, keytypes, bigram_scores, params)
    elif trigramtype == "skipgram":
        return get_skipgram_score(indices, keytypes, bigram_scores, params)
    elif trigramtype == "untypable":
        return dict(score=0, untypable=True)
    else:
        raise ValueError("Unknown trigram type")


def to_key_sequences(
    indices: list[int],
    keyseqtypes: list[HandOrKey],
) -> tuple[tuple[KeySeq], tuple[HandOrKey]]:
    return tuple(zip(*iterate_keyseqtypes_(indices, keyseqtypes)))  # type:ignore


def iterate_keyseqtypes_(
    indices: list[int],
    keyseqtypes: list[HandOrKey],
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


def to_skipgram_key_sequences(
    indices: list[int],
    keyseqtypes: list[HandOrKey],
) -> tuple[tuple[KeySeq, KeySeq], tuple[HandOrKey, HandOrKey]]:
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    if (
        keyseqtypes[0] != keyseqtypes[2]
        or keyseqtypes[0] == keyseqtypes[1]
        or keyseqtypes[0] not in HAND_TYPES
    ):
        raise ValueError("Only skipgrams are supported")
    keyseqtype1 = keyseqtypes[0]
    keyseqtype2 = keyseqtypes[1]
    keyseqs: tuple[KeySeq, KeySeq] = (indices[0], indices[2]), (indices[1],)
    return (keyseqs), (keyseqtype1, keyseqtype2)


def get_balanced_score(
    indices: list[int],
    keytypes: list[HandOrKey],
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
):
    """Get score for a balanced type of trigram."""
    key_sequences, key_seq_types = to_key_sequences(indices, keytypes)

    if len(key_sequences) != 2:
        raise ValueError("Only balanced trigrams are supported")

    try:
        bigram_score, unigram_score = _get_unigram_and_bigram_scores(
            key_sequences, key_seq_types, bigram_scores
        )

    except UntypableError:
        return dict(score=0, untypable=True)

    base_score = bigram_score + unigram_score
    score = (
        params.balanced_b_coeff * bigram_score + params.unigram_coeff * unigram_score
    )
    return dict(
        ngram1_score=bigram_score,
        ngram2_score=unigram_score,
        score=score,
        balanced_b_coeff=params.balanced_b_coeff,
        base_score=base_score,
        trigramtype="balanced",
    )


class UntypableError(RuntimeError): ...


def get_skipgram_score(
    indices: list[int],
    keytypes: list[HandOrKey],
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
) -> dict:
    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    key_sequences, key_seq_types = to_skipgram_key_sequences(indices, keytypes)

    try:
        bigram_score, unigram_score = _get_unigram_and_bigram_scores(
            key_sequences, key_seq_types, bigram_scores
        )

    except UntypableError:
        return dict(score=0, untypable=True)

    score = (
        params.skipgram_b_coeff * bigram_score + unigram_score * params.unigram_coeff
    )
    return dict(
        ngram1_score=bigram_score,
        ngram2_score=unigram_score,
        score=score,
        unigram_coeff=params.unigram_coeff,
        skipgram_b_coeff=params.skipgram_b_coeff,
        base_score=bigram_score + unigram_score,
        trigramtype="skipgram",
    )


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
    indices: list[int],
    hand: Hand,
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
    trigramtype: OnehandTrigramType,
):

    d = get_onehand_base_score(indices, hand, bigram_scores, params)

    if trigramtype == "rolling-easy":
        rolling = params.easy_rolling_coeff
    else:
        rolling = 1.0

    score = d["onehand_base_score"] * rolling
    return dict(
        **d,
        rolling=rolling,
        score=score,
        trigramtype=trigramtype,
    )


def get_onehand_base_score(
    indices: list[int],
    hand: Hand,
    bigram_scores: dict[KeySeq, float],
    params: TrigramModelParameters,
):

    extra = get_extra_multipliers(indices, hand, params)

    bigram1_score = bigram_scores[indices[0], indices[1]]
    bigram2_score = bigram_scores[indices[1], indices[2]]

    base_score = (3 / 4) * (bigram1_score + bigram2_score)

    onehand_base_score = extra["onehand_extra"] * base_score
    return extra | dict(
        ngram1_score=bigram1_score,
        ngram2_score=bigram2_score,
        base_score=base_score,
        onehand_base_score=onehand_base_score,
    )


def get_extra_multipliers(
    indices: list[int],
    hand: Hand,
    params: TrigramModelParameters,
):

    vert2u = get_vert2u_multiplier(indices, hand, coeff=params.vert2u_coeff)
    sft_multiplier = get_sft_multiplier(indices, hand, params.sft_coeff)
    sfb_in_onehand_multiplier = get_sfb_in_onehand_multiplier(
        indices,
        hand,
        params.sfb_in_onehand_coeff,
    )
    dirchange_multiplier = get_dirchange_multiplier(
        indices,
        hand,
        coeff=params.dirchange_coeff,
    )
    extra = vert2u * sft_multiplier * sfb_in_onehand_multiplier * dirchange_multiplier
    return dict(
        onehand_extra=extra,
        vert2u=vert2u,
        sft=sft_multiplier,
        sfb_in_onehand=sfb_in_onehand_multiplier,
        dirchange=dirchange_multiplier,
    )


def get_score(
    ngram: str,
    hands: Hands,
    params: TrigramModelParameters,
    bigram_scores: dict[KeySeq, float],
) -> dict:

    if len(ngram) == 3:
        return get_trigram_score(ngram, hands, params, bigram_scores=bigram_scores)
    raise NotImplementedError("Only trigrams are supported")


class TrigramScoreDict(typing.TypedDict):
    trigram_family_name: str
    trigram: str
    reference_trigram: str
    scaled_target_score: float
    scaled_estimated_score: float
    estimated_score: float
    estimated_score_details: dict


def iter_trigrams_scores(
    model_params: TrigramModelParameters,
    scoresets: TrigramScoreSets,
    hands: Hands,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
) -> Iterator[TrigramScoreDict]:
    """A generator function which yields a generator object (an iterator) which may be
    iterated over. Each returned item consists of the trigram scores and some additional
    information.

    Parameters
    ----------
    bigram_scores:
        Should be the bigram scores scaled from 1.0 to 5.0
    """

    def get_score_(trigram: str) -> dict:
        try:
            return get_trigram_score(
                trigram, hands, model_params, bigram_scores, mapping=mapping
            )
        except KeyError as err:
            raise RuntimeError(
                f"Trigram {trigram} could not be scored! The scores for indices {str(err)} not found in bigram scores! (see the Traceback for more details)"
            ) from err

    for reference_trigram_name, scoreset in scoresets:
        trigram_family_name = scoreset.ref_family
        ref_score_dct = get_score_(reference_trigram_name)
        ref_score = ref_score_dct["score"]

        for trigram, scaled_target_score in scoreset.scores.items():
            estimated_score_dct = get_score_(trigram)
            estimated_score = estimated_score_dct["score"]
            scaled_estimated_score = estimated_score / ref_score
            yield dict(
                trigram_family_name=trigram_family_name,
                trigram=trigram,
                reference_trigram=reference_trigram_name,
                scaled_target_score=scaled_target_score,
                scaled_estimated_score=scaled_estimated_score,
                estimated_score=estimated_score,
                estimated_score_details=estimated_score_dct,
            )


def get_scaled_bigram_scores(
    bigram_scores: dict[KeySeq, float],
    newmax: float,
    exponent: float = DEFAULT_BIGRAM_SCALING_EXPONENT,
) -> dict[KeySeq, float]:

    scaled_scores = dict()
    s = get_linear_scaling_function(oldmin=1, oldmax=5, newmin=1, newmax=newmax)

    for keyseq, score in bigram_scores.items():
        scaled_scores[keyseq] = s(score) ** exponent
    return scaled_scores


class TrigramScoreSet:

    scores: dict[str, float]
    """The scores relative to the reference trigram."""

    ref_family: str
    """The name of the family of the reference trigram ."""

    reference: str
    """The reference trigram."""

    def __init__(self, reference: str, family: str):
        self.reference = reference.upper()
        self.ref_family = family.upper()
        self.scores = dict()

    def add(self, trigram: str, score: float):
        self.scores[trigram.upper()] = score


class TrigramScoreSets:
    """Mapping of reference_trigram (keys) to TrigramScoreSet (values)"""

    def __init__(self, hands: Hands):
        self._hands = hands
        self._trigram_sets: dict[str, TrigramScoreSet] = dict()

    def add_trigram_set(self, trigram_set: TrigramScoreSet):
        self._trigram_sets[trigram_set.reference] = trigram_set

    def add_trigram(self, score: float, trigram: str, reference_trigram: str):

        self._trigram_sets[reference_trigram].add(trigram, score)

    def __getitem__(self, reference_trigram: str) -> TrigramScoreSet:
        return self._trigram_sets[reference_trigram]

    @classmethod
    def from_trigram_scores(
        cls, trigram_scoredict: dict[str, dict[str, float]], hands: Hands
    ) -> TrigramScoreSets:
        """Create TrigramScoreSets from a trigram score dictionary.

        Parameters
        ----------
        trigram_scoredict : dict[str, dict[str, float]]
            A dictionary where the keys are the reference trigrams and the values are
            dictionaries where the keys are the trigrams and the values are the scores.
        hands : Hands
            The hands data object (created from the config file).
        """
        trigram_sets = cls(hands)
        for reference_trigram, trigram_scores in trigram_scoredict.items():
            check_trigram(reference_trigram, hands)
            family = get_trigram_family(reference_trigram, hands)
            trigram_sets.add_trigram_set(TrigramScoreSet(reference_trigram, family))

            for trigram, score in trigram_scores.items():
                check_trigram(trigram, hands)
                trigram_sets.add_trigram(score, trigram, reference_trigram)
        return trigram_sets

    @classmethod
    def from_file(cls, file: str | Path, hands: Hands) -> TrigramScoreSets:
        trigram_scores = load_trigram_scores(file)
        return cls.from_trigram_scores(trigram_scores, hands)

    def __iter__(self) -> Iterator[tuple[str, TrigramScoreSet]]:
        return iter(self._trigram_sets.items())


def check_trigram(trigram: str, hands: Hands) -> None:
    indices, _ = hands.where(trigram)

    if len(indices) != 3:
        raise ValueError("Only trigrams are supported")
    elif len(set(indices)) != 3:
        raise ValueError(
            f"Trigrams should have three different indices! The trigram {trigram} contains indices {indices}!"
        )


def get_trigram_family(trigram: str, hands: Hands) -> str:
    indices, _ = hands.where(trigram)
    indices_tuple = typing.cast(tuple[int, int, int], tuple(indices))
    family = indices_to_family_name(indices_tuple, hands)
    return family


def load_trigram_scores(file) -> dict[str, dict[str, float]]:
    with open(file, "rb") as f:
        return tomllib.load(f)


def indices_to_family_name(indices: tuple[int, int, int], hands: Hands) -> str:
    """Gets the canonical name for a trigram family. That means that LEFT side inroll
    onehand for the trigram"""
    from granite_tools.scorer.trigram_creation import create_onehand_trigram

    return create_onehand_trigram(indices, "in", "Left", hands)[0]


def group_trigram_scores(
    scores: Iterable[TrigramScoreDict],
    group_sort_by: typing.Callable[[list[TrigramScoreDict]], float] | None = None,
) -> dict[str, list[TrigramScoreDict]]:
    """Group scores by trigram family name. The output dictionary keys will be the trigram
    set names. The values will be lists of dictionaries containing the scores. The order
    the scores are added to the list is the same as the order they are yielded by the
    input iterable. Optionally, sorts the groups using a group key function.

    Parameters
    ----------
    group_sort_by : typing.Callable[[TrigramScoreDict], float] | None
        If given, is a function that takes a TrigramScoreDict and returns a float. This
        is used to sort the trigram groups."""
    out = defaultdict(list)
    for item in scores:
        out[item["trigram_family_name"]].append(item)
    if group_sort_by is None:
        return out
    return sort_trigram_score_groups(out, key=group_sort_by)


def sort_trigram_score_groups(
    groups: dict[str, list[TrigramScoreDict]],
    key: typing.Callable[[list[TrigramScoreDict]], float],
) -> dict[str, list[TrigramScoreDict]]:
    """Sorts the trigram score groups by the given key function."""
    return {
        name: group for name, group in sorted(groups.items(), key=lambda x: key(x[1]))
    }


def max_abs_error(dcts: Sequence[TrigramScoreDict]) -> float:
    return max(
        [abs(d["scaled_target_score"] - d["scaled_estimated_score"]) for d in dcts]
    )


def average_abs_error(dcts: Sequence[TrigramScoreDict]) -> float:
    return sum(
        [abs(d["scaled_target_score"] - d["scaled_estimated_score"]) for d in dcts]
    ) / len(dcts)


class TrigramModelParameters(BaseModel):

    model_config = ConfigDict(extra="forbid")  # helps to spot typos.

    # Optimized with a minimizer
    vert2u_coeff: float = DEFAULT_VERT2U_COEFF
    dirchange_coeff: float = DEFAULT_DIRCHANGE_COEFF
    balanced_b_coeff: float = DEFAULT_BALANCED_B_COEFF
    unigram_coeff: float = DEFAULT_UNIGRAM_COEFF
    skipgram_b_coeff: float = DEFAULT_SKIPGRAM_B_COEFF
    easy_rolling_coeff: float = DEFAULT_EASY_ROLLING_COEFF
    bigram_raw_range_max: float = DEFAULT_BIGRAM_RAW_RANGE_MAX
    bigram_scaling_exponent: float = DEFAULT_BIGRAM_SCALING_EXPONENT

    # Non-optimized (manually tuned)
    sft_coeff: float = DEFAULT_SFT_COEFF
    sfb_in_onehand_coeff: float = DEFAULT_SFB_IN_ONEHAND_COEFF

    @classmethod
    def from_config(cls, config: Config) -> TrigramModelParameters:
        return cls(
            vert2u_coeff=(
                config.vert2u_coeff
                if config.vert2u_coeff is not None
                else DEFAULT_VERT2U_COEFF
            ),
            dirchange_coeff=(
                config.dirchange_coeff
                if config.dirchange_coeff is not None
                else DEFAULT_DIRCHANGE_COEFF
            ),
            balanced_b_coeff=(
                config.balanced_b_coeff
                if config.balanced_b_coeff is not None
                else DEFAULT_BALANCED_B_COEFF
            ),
            unigram_coeff=(
                config.unigram_coeff
                if config.unigram_coeff is not None
                else DEFAULT_UNIGRAM_COEFF
            ),
            skipgram_b_coeff=(
                config.skipgram_b_coeff
                if config.skipgram_b_coeff is not None
                else DEFAULT_SKIPGRAM_B_COEFF
            ),
            easy_rolling_coeff=(
                config.easy_rolling_coeff
                if config.easy_rolling_coeff is not None
                else DEFAULT_EASY_ROLLING_COEFF
            ),
            sft_coeff=(
                config.sft_coeff if config.sft_coeff is not None else DEFAULT_SFT_COEFF
            ),
            sfb_in_onehand_coeff=(
                config.sfb_in_onehand_coeff
                if config.sfb_in_onehand_coeff is not None
                else DEFAULT_SFB_IN_ONEHAND_COEFF
            ),
            bigram_raw_range_max=(
                config.bigram_raw_range_max
                if config.bigram_raw_range_max is not None
                else DEFAULT_BIGRAM_RAW_RANGE_MAX
            ),
            bigram_scaling_exponent=(
                config.bigram_scaling_exponent
                if config.bigram_scaling_exponent is not None
                else DEFAULT_BIGRAM_SCALING_EXPONENT
            ),
        )

    @classmethod
    def from_tuple(cls, x: tuple[float, ...]) -> TrigramModelParameters:
        return cls(
            vert2u_coeff=x[0],
            dirchange_coeff=x[1],
            balanced_b_coeff=x[2],
            unigram_coeff=x[3],
            skipgram_b_coeff=x[4],
            easy_rolling_coeff=x[5],
            bigram_raw_range_max=x[6],
            bigram_scaling_exponent=x[7],
        )

    def as_tuple(self, only_model_params: bool = True) -> tuple[float, ...]:

        model_params = (
            self.vert2u_coeff,
            self.dirchange_coeff,
            self.balanced_b_coeff,
            self.unigram_coeff,
            self.skipgram_b_coeff,
            self.easy_rolling_coeff,
            self.bigram_raw_range_max,
            self.bigram_scaling_exponent,
        )
        if only_model_params:
            return model_params
        return model_params + (
            self.sft_coeff,
            self.sfb_in_onehand_coeff,
        )


def get_trigram_params_error_fun(
    scoresets: TrigramScoreSets,
    hands: Hands,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
):
    # Calculate this mapping only once as it's always the same. About 85% speed improvement.
    mapping = mapping or (
        get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
        if hands.config.easy_rolling_trigrams is not None
        else None
    )

    def scorefunc(x, *_):
        # Calculates RMSE
        all_scores_func = get_all_errors_func(
            scoresets, hands, bigram_scores, mapping=mapping
        )
        scoredct = all_scores_func(x, *_)
        squared_sum = 0
        for score in scoredct.values():
            squared_sum += score**2
        return (squared_sum / len(scoredct)) ** 0.5

    return scorefunc


def get_all_errors_func(
    scoresets: TrigramScoreSets,
    hands: Hands,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
):
    """Create a function that returns a list of scores for all trigrams in the scoresets"""
    mapping = mapping or (
        get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
        if hands.config.easy_rolling_trigrams is not None
        else None
    )

    def scorefunc(x, *_):
        model_params = TrigramModelParameters.from_tuple(x)

        scores = dict()
        for d in iter_trigrams_scores(
            model_params, scoresets, hands, bigram_scores, mapping=mapping
        ):
            score = d["scaled_target_score"] - d["scaled_estimated_score"]
            scores[d["trigram"]] = score

        return scores

    return scorefunc


def get_initial_params_and_bounds() -> list[float]:
    """Create initial params for optimization"""
    x0 = [
        1.4,  # vert2u_coeff
        1.5,  # dirchange_coeff
        0.5,  # balanced_b_coeff
        0.6,  # unigram_coeff
        1.1,  # skipgram_b_coeff
        0.6,  # easy_rolling_coeff
        40,  # bigram_raw_range_max
        0.3,  # bigram_scaling_exponent
    ]

    bounds = [
        (0.1, 5.0),  # vert2u_coeff
        (0.1, 5.0),  # dirchange_coeff
        (0.1, 1.0),  # balanced_b_coeff
        (0.05, 1.0),  # unigram_coeff
        (0.1, 2.0),  # skipgram_b_coeff
        (0.1, 1.0),  # easy_rolling_coeff
        (1.1, 400),  # bigram_raw_range_max
        (0.1, 0.7),  # bigram_scaling_exponent
    ]
    return x0, bounds


def optimize_parameters(
    scorefunc: typing.Callable[[list[float]], float],
    x0: list[float],
    bounds: list[tuple[float, float]],
    config: Config,
) -> tuple[float, ...]:

    res = minimize(
        scorefunc,
        x0=np.array(x0),
        bounds=bounds,
        method="Nelder-Mead",
        args=(config,),
        options=dict(maxiter=None, disp=True),
    )  # type: ignore

    return tuple(float(x) for x in res.x)


if __name__ == "__main__":
    from granite_tools.scorer.bigram_scores import load_bigram_and_unigram_scores

    bigram_and_unigram_scores = load_bigram_and_unigram_scores(sys.argv[1])

    config = read_config(sys.argv[2])
    params = TrigramModelParameters.from_config(config)
    hands = get_hands_data(config)
    while True:
        ngram = input("type ngram: ")
        try:
            scoredct = get_score(ngram, hands, params, bigram_and_unigram_scores)
            print(scoredct)
            print(f'Score of "{ngram}" is {scoredct["score"]}')
        except Exception as e:
            print(e)
