"""Reusable types for the granite-tools apps."""

from __future__ import annotations

import typing
from enum import Enum, auto
from typing import Literal, TypedDict

HandType = Literal["Left", "Right"]
HandOrKey = Literal["Left", "Right", "Untypable"]
"""Hand or the key type."""

HAND_TYPES = tuple(typing.get_args(HandType))

KeySeq = tuple[int, ...]

BigramOrUnigram = Literal["unigram", "bigram"]


class BigramScoreDict(TypedDict):
    """A dictionary containing information about a bigram score."""

    key_indices: KeySeq
    """The indices of the keys on the keyboard (defined in config file)."""

    type: BigramOrUnigram
    """The type of the ngram. Bigram score files contain also data for unigrams."""

    score: float
    """The score of the bigram (or unigram)."""

    rank: int
    """The rank of the ngram (bigram or unigram) in the list of ngrams. 1 means the 
    easiest to type (and lowest score)."""

    rank_type: int
    """The rank of the unigram or bigram within the ngram type."""

    __comment__left: str
    __comment__right: str


class OrderedStrEnum(Enum):
    def __lt__(self, other):
        if isinstance(other, OrderedStrEnum):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OrderedStrEnum):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OrderedStrEnum):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OrderedStrEnum):
            return self.value >= other.value
        return NotImplemented


class FingerType(OrderedStrEnum):
    # From least to most effort
    T = auto()  # thumb
    I = auto()  # index # noqa: E741
    M = auto()  # middle
    R = auto()  # ring
    P = auto()  # pinky

    @classmethod
    def from_str(cls, finger: str) -> FingerType:
        return {
            "t": cls.T,
            "i": cls.I,
            "m": cls.M,
            "r": cls.R,
            "p": cls.P,
        }[finger]


Ring = FingerType.R
Index = FingerType.I
Middle = FingerType.M
Pinky = FingerType.P
Thumb = FingerType.T


class RepeatType(OrderedStrEnum):
    # From least to most effort
    REP = auto()  # Repeated key
    SFS = auto()  # Single Finger Skipgram
    SFB = auto()  # Single Finger Bigram
    RSFT = auto()  # Single Finger Trigram with a repeat
    SFT = auto()  # Single Finger Trigram


class RowDiffType(OrderedStrEnum):
    # From least to most effort
    # 1u: 1 unit differece in rows
    # 2u: 2 unit differece in rows
    RowDiff2u = auto()
    # These others consists of "scissor" type of movement
    # and other non-ergonomic movements caused by row difference.
    MiddleBelowIndex2u = auto()  # two positions
    MiddleBelowPinky1u = auto()
    IndexBelowPinky2u = auto()
    MiddleBelowRing2u = auto()
    PinkyBelowRing2u = auto()
    MiddleBelowPinky2u = auto()
    RingBelowPinky1u = auto()
    RingBelowPinky2u = auto()


class DirectionType(OrderedStrEnum):
    """Directions track the some unfomfortable horizontal movements. From bigrams, they
    catch few outward and inwards rolls (involving pinky but no index). From trigrams,
    they additionally catch different types of redirects.

    Here are all trigrams with three different fingers used:

        p  r  m  i
       ------------
    1:  1  2  3        in (pr)
    2:  1  2     3     in (pr)
    3:  1  3  2        redir4
    4:  1  3     2     redir2
    5:  1     2  3     in (pm)
    6:  1     3  2     redir2
    7:  2  1  3        redir4
    8:  2  1     3     redir3
    9:  2  3  1        redir4
    10: 2  3     1     redir3
    11: 2     1  3     redir3
    12: 2     3  1     redir3
    13: 3  1  2        redir4
    14: 3  1     2     redir2
    15: 3  2  1        out (rp)
    16: 3  2     1     out (rp)
    17: 3     1  2     redir2
    18: 3     2  1     out (mp)
    19:    1  2  3
    20:    1  3  2     redir1
    21:    2  1  3     redir1
    22:    2  3  1     redir1
    23:    3  1  2     redir1
    24     3  2  1

    Redirect levels (easiest to hardest)
    1: No pinky invoved (easiest). Score (a.u.): 0.4
    2: Index in the middle. Score (a.u): 3.1
    3: Index not in the middle. Score: (a.u): 10
    4: Index not included. Score: (a.u) 25

    In/Out (easiest to most difficult)
    * in (pm): Pinky->Middle. Score (a.u):  0.8
    * in (pr): Pinky->Ring. Score (a.u):  1.5
    * out (mp): Middle -> Pinky. Score (a.u.): 2.5
    * out (rp): Ring -> Pinky. Score (a.u.): 7
    """

    Redirect1 = 0.4
    InwardsPinkyMiddle = 0.8
    InwardsPinkyRing = 1.5
    OutwardsMiddlePinky = 2.5
    Redirect2 = 3.1
    OutwardsRingPinky = 7
    Redirect3 = 10
    Redirect4 = 25


TrigramMainType = Literal["balanced", "alternating", "easy-rolling", "onehand"]

TRIGRAM_MAIN_TYPES = list(typing.get_args(TrigramMainType))


Vert2uPenaltyException = tuple[int, list[int]]


# For configuration: vert2u_penalties
class Vert2uPenaltyConfig(TypedDict):
    exceptions: list[Vert2uPenaltyException]
