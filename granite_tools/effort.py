from __future__ import annotations

import itertools
import typing
from enum import Enum, auto
from itertools import zip_longest
from typing import ClassVar, Literal, TypedDict, cast

from pydantic import BaseModel, Field, field_validator
from rich.text import Text

if typing.TYPE_CHECKING:
    from granite_tools.config import Config

HandType = Literal["Left", "Right"]


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


def get_rowdiff_for_bigram(
    finger1: FingerType, row1: int, finger2: FingerType, row2: int
) -> RowDiffType | None:
    """Gets a row difference type for a bigram (sequence of two keys). The rows are assumed to grow from top to bottom;
    larger row numbers are below smaller row numbers."""

    finger_rows = (finger1, row1), (finger2, row2)
    lower_finger, _ = max(*finger_rows, key=lambda x: x[1])
    higher_finger, _ = min(*finger_rows, key=lambda x: x[1])
    diff = abs(row1 - row2)

    if diff == 0:
        return None

    if lower_finger == FingerType.M and higher_finger == FingerType.I:
        if diff == 2:
            return RowDiffType.MiddleBelowIndex2u
    elif lower_finger == FingerType.M and higher_finger == FingerType.P:
        if diff == 1:
            return RowDiffType.MiddleBelowPinky1u
        if diff == 2:
            return RowDiffType.MiddleBelowPinky2u
    elif lower_finger == FingerType.P and higher_finger == FingerType.R:
        if diff == 2:
            return RowDiffType.PinkyBelowRing2u
    elif lower_finger == FingerType.R and higher_finger == FingerType.P:
        if diff == 1:
            return RowDiffType.RingBelowPinky1u
        if diff == 2:
            return RowDiffType.RingBelowPinky2u
    elif lower_finger == FingerType.I and higher_finger == FingerType.P:
        if diff == 2:
            return RowDiffType.IndexBelowPinky2u
    elif lower_finger == FingerType.M and higher_finger == FingerType.R:
        if diff == 2:
            return RowDiffType.MiddleBelowRing2u
    elif diff == 2:
        return RowDiffType.RowDiff2u
    return None


def get_direction_for_bigram(
    finger1: FingerType,
    finger2: FingerType,
) -> DirectionType | None:
    """Gets a direction type or a bigram (sequence of two keys)."""

    if finger1 == FingerType.P and finger2 == FingerType.R:
        return DirectionType.InwardsPinkyRing
    elif finger1 == FingerType.P and finger2 == FingerType.M:
        return DirectionType.InwardsPinkyMiddle
    elif finger1 == FingerType.M and finger2 == FingerType.P:
        return DirectionType.OutwardsMiddlePinky
    elif finger1 == FingerType.R and finger2 == FingerType.P:
        return DirectionType.OutwardsRingPinky
    return None


def get_direction_for_trigram(
    finger1: FingerType,
    finger2: FingerType,
    finger3: FingerType,
) -> DirectionType | None:
    """Gets a direction type or a bigram (sequence of two keys)."""

    fingers = finger1, finger2, finger3
    if (
        fingers == (Ring, Index, Middle)
        or fingers == (Middle, Ring, Index)
        or fingers == (Index, Ring, Middle)
        or fingers == (Middle, Index, Ring)
    ):
        return DirectionType.Redirect1
    elif (
        fingers == (Middle, Index, Pinky)
        or fingers == (Ring, Index, Pinky)
        or fingers == (Pinky, Index, Middle)
        or fingers == (Pinky, Index, Ring)
    ):
        return DirectionType.Redirect2
    elif (
        fingers == (Index, Pinky, Middle)
        or fingers == (Middle, Pinky, Index)
        or fingers == (Index, Pinky, Ring)
        or fingers == (Ring, Pinky, Index)
    ):
        return DirectionType.Redirect3
    elif (
        fingers == (Pinky, Middle, Ring)
        or fingers == (Middle, Pinky, Ring)
        or fingers == (Ring, Pinky, Middle)
        or fingers == (Ring, Middle, Pinky)
    ):
        return DirectionType.Redirect4

    bigram1 = get_direction_for_bigram(finger1, finger2)
    bigram2 = get_direction_for_bigram(finger2, finger3)

    if bigram1 is None:
        return bigram2
    elif bigram2 is None:
        return bigram1
    return max(bigram1, bigram2)


class FingerType(OrderedStrEnum):
    # From least to most effort
    T = auto()  # thumb
    I = auto()  # index
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


class Hand(BaseModel):
    hand: HandType
    symbols: dict[int, str]
    """Keys: key indices. Values: symbols on keyboard."""

    fingers: dict[int, str] = Field(default_factory=dict)
    """For calculaing if somethign is SFB."""

    key_categories: dict[int, str] = Field(default_factory=dict)
    """For visualization. Keys: key indices. Values: symbols for fingers."""

    colors: dict[int, str] = Field(default_factory=dict)
    """For visualization. Keys: key indices. Values: color names (understood by Rich)"""

    matrix_positions: dict[int, tuple[int, int]] = Field(default_factory=dict)
    """Keys: key indices. Values: (column, row) positions in the matrix."""

    @field_validator("symbols", mode="before")
    def sort_dict(cls, v):
        return dict(sorted(v.items(), key=lambda item: item[0]))

    def get_symbols(self, key_seq: tuple[int, ...] | None, fallback="") -> str:
        if key_seq is None:
            return fallback
        # Get a symbol
        symbols = ""
        for key_idx in key_seq:
            if key_idx not in self.symbols:
                return fallback
            symbols += self.symbols[key_idx]
        return symbols

    def get_finger(self, key_idx: int) -> FingerType | None:
        """Returns the finger for a given key index."""
        f = self.fingers.get(key_idx)
        if not f:
            return None
        return FingerType.from_str(f)

    def get_repeats_tuple(
        self, key_seq: tuple[int, ...]
    ) -> tuple[RepeatType, FingerType] | None:
        """Returns the type of repeat in the key sequence. If it's not a repeat, returns
        None."""
        if not self.fingers:
            return None

        if len(key_seq) == 0:
            return None
        elif len(key_seq) > 3:
            raise ValueError("Only supports up to trigrams")
        elif len(key_seq) == 1:
            return None
        f0 = self.get_finger(key_seq[0])
        f1 = self.get_finger(key_seq[1])

        if len(key_seq) == 2:
            if f0 == f1 and f0 is not None:
                if key_seq[0] == key_seq[1]:
                    return RepeatType.REP, f0
                return RepeatType.SFB, f0
            else:
                return None
        # length is 3
        f2 = self.get_finger(key_seq[2])
        if f2 is None or f0 is None:
            return None
        elif f0 == f2 and f0 != f1:
            return RepeatType.SFS, f0
        elif f0 == f2 and f0 == f1:
            if key_seq[0] == key_seq[1] and key_seq[1] == key_seq[2]:
                return RepeatType.REP, f0
            if key_seq[0] == key_seq[1] or key_seq[1] == key_seq[2]:
                return RepeatType.RSFT, f0
            return RepeatType.SFT, f0
        elif f0 == f1:
            if key_seq[0] == key_seq[1]:
                return RepeatType.REP, f0
            return RepeatType.SFB, f0
        elif f1 == f2:
            return RepeatType.SFB, f2
        return None

    def get_rowdiff(
        self, key_seq: tuple[int, ...]
    ) -> tuple[RowDiffType] | tuple[RowDiffType, RowDiffType] | None:

        if len(key_seq) == 1:
            return None
        elif len(key_seq) > 3:
            raise ValueError("Only supports up to trigrams")

        finger1 = self.get_finger(key_seq[0])
        finger2 = self.get_finger(key_seq[1])

        if finger1 is None or finger2 is None:
            return None

        coord1 = self.matrix_positions.get(key_seq[0])
        coord2 = self.matrix_positions.get(key_seq[1])

        if coord1 is None or coord2 is None:
            return None

        row1, row2 = coord1[1], coord2[1]

        diff1 = get_rowdiff_for_bigram(finger1, row1, finger2, row2)
        if len(key_seq) == 2:
            if diff1 is None:
                return None
            return (diff1,)

        # 3 keys
        finger3 = self.get_finger(key_seq[2])
        if finger3 is None:
            return None
        coord3 = self.matrix_positions.get(key_seq[2])
        if coord3 is None:
            return None
        row3 = coord3[1]
        diff2 = get_rowdiff_for_bigram(finger2, row2, finger3, row3)

        if diff2 is None:
            if diff1 is None:
                return None
            return (diff1,)
        if diff1 is None:
            return (diff2,)
        return (diff1, diff2)

    def get_fingers_str(self, key_seq: tuple[int, ...]) -> Text | None:
        """Gets the fingers string for presentation for a given key indices.

        Parameters
        ----------
        key_seq : tuple[int, ...]
            The key indices. For example (5,0,1)

        Returns
        -------
        list[str]
            The fingers string for presentation.
        """
        if not self.key_categories:
            return None

        fingers: list[str | tuple[str, str]] = []
        for key_idx in key_seq:
            if key_idx not in self.key_categories:
                return None
            finger = self.key_categories[key_idx]
            color = self.colors.get(key_idx)
            if not color:
                fingers.append(finger)
            else:
                fingers.append((finger, color))
        return Text.assemble(*fingers)

    def get_direction(self, key_seq: tuple[int, ...]) -> DirectionType | None:
        if len(key_seq) <= 1:
            return None
        elif len(key_seq) > 3:
            raise ValueError("Only supports up to trigrams")
        finger1 = self.get_finger(key_seq[0])
        finger2 = self.get_finger(key_seq[1])
        if finger1 is None or finger2 is None:
            return None
        if len(key_seq) == 2:
            return get_direction_for_bigram(finger1, finger2)
        finger3 = self.get_finger(key_seq[2])
        if finger3 is None:
            return None
        return get_direction_for_trigram(finger1, finger2, finger3)


class Hands(BaseModel):
    left: Hand
    right: Hand

    def get_symbols(
        self, hand: HandType, key_seq: tuple[int, ...] | None, fallback=""
    ) -> str:
        handobj = getattr(self, hand.lower())
        return handobj.get_symbols(key_seq, fallback)

    def get_symbols_text(
        self,
        hand: HandType,
        key_seq: tuple[int, ...] | None,
        fallback="",
        center: int | None = None,
    ) -> Text:
        symbols = self.get_symbols(hand, key_seq, fallback)
        color = "sky_blue1" if hand.lower() == "left" else "light_pink1"
        if center is not None:
            symbols = symbols.center(center)
        return Text.assemble((symbols, f"bold {color}"))

    def get_fingers_str(self, key_seq: tuple[int, ...]) -> Text | str:
        """Returns a 'combined' fingers string. If both hands have the same fingers
        string, returns that (once). If they are different, returns both strings.
        If only one of them exists, returns that."""
        fingers_left = self.left.get_fingers_str(key_seq)
        fingers_right = self.right.get_fingers_str(key_seq)
        if not fingers_left and not fingers_right:
            return ""
        if not fingers_left:
            if isinstance(fingers_right, Text):
                return fingers_right
            return ""
        if not fingers_right:
            return fingers_left
        if fingers_left == fingers_right:
            return fingers_left
        return Text.assemble(fingers_left, "  ", fingers_right)

    def get_repeats_tuple(
        self, key_seq: tuple[int, ...]
    ) -> tuple[RepeatType, FingerType] | None:
        """Returns the type of repeat in the key sequence. If both left and right have different
        repeats, returns the one with takes the most effort. (first order by RepeatType
        and then by FingerType)"""
        left_repeats = self.left.get_repeats_tuple(key_seq)
        right_repeats = self.right.get_repeats_tuple(key_seq)

        if left_repeats == right_repeats:
            return left_repeats
        if left_repeats is None and right_repeats is None:
            return None
        if left_repeats is None:
            return right_repeats
        if right_repeats is None:
            return left_repeats
        return max(left_repeats, right_repeats)

    repeat_colors: ClassVar = {
        RepeatType.REP: "gray",
        RepeatType.SFS: "royal_blue1",
        RepeatType.SFB: "#e36f42",
        RepeatType.RSFT: "deep_pink3",
        RepeatType.SFT: "deep_pink2",
    }

    finger_colors: ClassVar = {
        FingerType.P: "deep_pink3",
        FingerType.R: "red",
        FingerType.M: "blue1",
        FingerType.I: "#069606",
        FingerType.T: "#7d807d",
    }

    def get_repeats_text(self, key_seq: tuple[int, ...]) -> Text:
        """Returns a text object that indicates if the key sequence is a repeat."""
        repeat_tuple = self.get_repeats_tuple(key_seq)
        if not repeat_tuple:
            return Text("")
        repeat_type, finger_type = repeat_tuple

        return Text.assemble(
            (repeat_type.name, self.repeat_colors[repeat_type]),
            "(",
            (finger_type.name.lower(), self.finger_colors[finger_type]),
            ")",
        )

    def get_rowdiff(
        self, key_seq: tuple[int, ...]
    ) -> tuple[RowDiffType] | tuple[RowDiffType, RowDiffType] | None:
        """Returns the type of row difference in the key sequence. If both left and
        right hand have different row difference types, returns the one with the highest
        effort rowdiff type."""
        left_rowdiff = self.left.get_rowdiff(key_seq)
        right_rowdiff = self.right.get_rowdiff(key_seq)

        if left_rowdiff == right_rowdiff:
            return left_rowdiff
        if left_rowdiff is None and right_rowdiff is None:
            return None
        if left_rowdiff is None:
            return right_rowdiff
        if right_rowdiff is None:
            return left_rowdiff
        return max(left_rowdiff, right_rowdiff, key=max)

    rowdiff_colors: ClassVar = {
        RowDiffType.RowDiff2u: "gray",
        RowDiffType.RingBelowPinky2u: "deep_pink3",
        RowDiffType.RingBelowPinky1u: "deep_pink2",
        RowDiffType.MiddleBelowPinky2u: "purple",
        RowDiffType.PinkyBelowRing2u: "red",
        RowDiffType.MiddleBelowRing2u: "dark_orange",
        RowDiffType.IndexBelowPinky2u: "yellow1",
        RowDiffType.MiddleBelowPinky1u: "spring_green3",
        RowDiffType.MiddleBelowIndex2u: "dodger_blue2",
    }
    rowdiff_names: ClassVar = {
        RowDiffType.RowDiff2u: "2u",
        RowDiffType.RingBelowPinky2u: "rp2u",
        RowDiffType.RingBelowPinky1u: "rp1u",
        RowDiffType.MiddleBelowPinky2u: "mp2u",
        RowDiffType.PinkyBelowRing2u: "pr2u",
        RowDiffType.MiddleBelowRing2u: "mr2u",
        RowDiffType.IndexBelowPinky2u: "ip2u",
        RowDiffType.MiddleBelowPinky1u: "mp1u",
        RowDiffType.MiddleBelowIndex2u: "mi2u",
    }

    def get_rowdiff_text(self, key_seq: tuple[int, ...]) -> Text:
        """The row difference as renderable, if any. If the key_seq (trigram) has multiple
        row differences, returns both of them in a single Text object."""
        rowdiff = self.get_rowdiff(key_seq)
        if not rowdiff:
            return Text("")

        if len(rowdiff) == 1:
            return Text.assemble(
                (
                    self.rowdiff_names[rowdiff[0]],
                    self.rowdiff_colors.get(rowdiff[0], "gray"),
                )
            )
        return Text.assemble(
            (
                self.rowdiff_names[rowdiff[0]],
                self.rowdiff_colors.get(rowdiff[0], "gray"),
            ),
            " ",
            (
                self.rowdiff_names[rowdiff[1]],
                self.rowdiff_colors.get(rowdiff[1], "gray"),
            ),
        )

    def get_direction(self, key_seq: tuple[int, ...]) -> DirectionType | None:
        """Returns the direction type for the key sequence."""
        direction_left = self.left.get_direction(key_seq)
        direction_right = self.right.get_direction(key_seq)

        if direction_left == direction_right:
            return direction_left
        if direction_left is None:
            return direction_right
        if direction_right is None:
            return direction_left
        return max(direction_left, direction_right)

    direction_colors: ClassVar = {
        DirectionType.Redirect4: "deep_pink3",
        DirectionType.Redirect3: "red",
        DirectionType.OutwardsRingPinky: "dark_orange3",
        DirectionType.Redirect2: "#edd928",
        DirectionType.OutwardsMiddlePinky: "#56e04c",
        DirectionType.InwardsPinkyRing: "#46e3db",
        DirectionType.InwardsPinkyMiddle: "#3b6bdb",
        DirectionType.Redirect1: "#7d807d",
    }
    direction_names: ClassVar = {
        DirectionType.Redirect4: "redir4",
        DirectionType.Redirect3: "redir3",
        DirectionType.OutwardsRingPinky: "out(rp)",
        DirectionType.Redirect2: "redir2",
        DirectionType.OutwardsMiddlePinky: "out(mp)",
        DirectionType.InwardsPinkyRing: "in(pr)",
        DirectionType.InwardsPinkyMiddle: "in(pm)",
        DirectionType.Redirect1: "redir1",
    }

    def get_direction_text(self, key_seq: tuple[int, ...]) -> Text:
        """The direction as renderable, if any. If the key_seq (trigram) has multiple
        directions, return the most difficult one."""
        direction = self.get_direction(key_seq)
        if direction is None:
            return Text("")
        return Text.assemble(
            (
                self.direction_names.get(direction, "?"),
                self.direction_colors.get(direction, "gray"),
            )
        )


def get_hands_data(config: Config) -> Hands:

    class HandData(TypedDict):
        symbols: dict[int, str]
        fingers: dict[int, str]
        key_categories: dict[int, str]
        colors: dict[int, str]
        matrix_positions: dict[int, tuple[int, int]]

    class HandsData(TypedDict):
        Left: HandData
        Right: HandData

    hands: HandsData = {
        "Left": {
            "symbols": {},
            "fingers": {},
            "key_categories": {},
            "colors": {},
            "matrix_positions": {},
        },
        "Right": {
            "symbols": {},
            "fingers": {},
            "key_categories": {},
            "colors": {},
            "matrix_positions": {},
        },
    }

    nonematrix = [[None for _ in sublist] for sublist in config.symbols]
    finger_matrix = config.finger_matrix or nonematrix
    key_category_matrix = config.key_category_matrix or nonematrix
    color_matrix = config.color_matrix or nonematrix
    matrix_positions = config.matrix_positions or nonematrix
    error = "__ERROR_FILL_VALUE__"
    default_color = "white"

    row_iterator = zip_longest(
        config.hands,
        config.symbols,
        config.key_indices,
        finger_matrix,
        key_category_matrix,
        color_matrix,
        matrix_positions,
        fillvalue=error,
    )

    for row_data in row_iterator:
        if error in row_data:
            raise ValueError("Invalid config! One block has more rows than others")
        for cell_data in zip_longest(
            *row_data,
            fillvalue=error,
        ):  # type: ignore
            if error in cell_data:
                raise ValueError(
                    "Invalid config! One row is longer or shorter than others"
                )
            hand, symbol, index, finger, key_category, colorname, matrix_position = (
                cell_data
            )
            index_int: int = cast(int, int(index))  # type: ignore
            colorname_str: str = cast(str, str(colorname))  # type: ignore

            if config.color_mapping:
                color = config.color_mapping.get(colorname_str, default_color)
            else:
                color = None

            if hand not in ["Left", "Right"]:
                raise ValueError(f"Invalid hand: {hand}")
            hand = typing.cast(Literal["Left", "Right"], hand)

            dct = hands[hand]  # type: ignore # (no idea why mypy still complains.)

            dct["symbols"][index_int] = str(symbol)
            if finger:
                dct["fingers"][index_int] = str(finger)
            if color:
                dct["colors"][index_int] = str(color)
            if key_category:
                dct["key_categories"][index_int] = str(key_category)
            if matrix_position:
                if not isinstance(matrix_position, tuple) and len(matrix_position) == 2:  # type: ignore
                    raise ValueError(f"Invalid matrix position: {matrix_position}")
                if not all(isinstance(i, int) for i in matrix_position):  # type: ignore
                    raise ValueError(f"Invalid matrix position: {matrix_position}")
                matrix_position = cast(tuple[int, int], matrix_position)
                dct["matrix_positions"][index_int] = matrix_position

    return Hands(
        left=Hand(
            hand="Left",
            symbols=hands["Left"]["symbols"],
            key_categories=hands["Left"]["key_categories"],
            fingers=hands["Left"]["fingers"],
            colors=hands["Left"]["colors"],
            matrix_positions=hands["Left"]["matrix_positions"],
        ),
        right=Hand(
            hand="Right",
            symbols=hands["Right"]["symbols"],
            key_categories=hands["Right"]["key_categories"],
            fingers=hands["Right"]["fingers"],
            colors=hands["Right"]["colors"],
            matrix_positions=hands["Right"]["matrix_positions"],
        ),
    )


def create_permutations(
    left: Hand, right: Hand, sequence_lengths: tuple[int, ...] = (1, 2)
) -> list[tuple[int, ...]]:
    """This creates permutations of given sequence lengths that can be typed with at
    least one of the hands. The returned list contains tuples of key indices."""
    key_indices = get_union_of_keys(left, right)

    permutations_lst = []
    for seq_length in sequence_lengths:
        for seq in itertools.product(key_indices, repeat=seq_length):

            if not permutation_is_typable(left, right, seq):
                continue
            permutations_lst.append(seq)

    return permutations_lst


def permutation_is_typable(
    left: Hand, right: Hand, permutation: tuple[int, ...]
) -> bool:
    """Check if permutation is typable with at least one hand"""
    for hand in (left, right):
        can_be_typed = True

        for key in permutation:
            if key not in hand.symbols:
                can_be_typed = False
                break

        if can_be_typed:
            return True

    return False


def get_union_of_keys(left: Hand, right: Hand) -> list[int]:
    """Gets the union of key indices from both hands."""
    return sorted(set(left.symbols.keys()) | set(right.symbols.keys()))  # type: ignore
