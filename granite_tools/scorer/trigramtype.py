from __future__ import annotations

import typing

import numpy as np

from granite_tools.app_types import FingerType, TrigramMainType

if typing.TYPE_CHECKING:
    from typing import Iterable

    from granite_tools.app_types import OnehandTrigramType, TrigramType
    from granite_tools.hands import Hand, Hands

    Coord = tuple[int, int]
    TrigramShapeCoords = tuple[Coord, Coord, Coord]


Indices = tuple[int, int, int]
EasyRollingTrigramsMap = dict[Indices, str]


def get_trigram_type(
    trigram: str, hands: Hands, mapping: EasyRollingTrigramsMap | None = None
) -> TrigramType:
    """Get a trigram type. The trigram mapping can be provided as an argument which
    makes the trigram type checking much faster."""
    trigramtype: TrigramType
    _, keytypes = hands.where(trigram)
    try:
        trigram_main_type = hands.get_trigram_type(keytypes)
    except Exception:
        raise RuntimeError(f"Trigram type not recognized for '{trigram}'")
    if trigram_main_type == TrigramMainType.BALANCED:
        trigramtype = "balanced"
    elif trigram_main_type == TrigramMainType.SKIPGRAM:
        trigramtype = "skipgram"
    elif trigram_main_type == TrigramMainType.ONEHAND:
        if mapping is None and hands.config.easy_rolling_trigrams is not None:
            mapping = get_easy_rolling_type_mapping(
                hands.config.easy_rolling_trigrams, hands
            )
        onehand_type = get_onehand_trigram_type(trigram, mapping, hands)
        if onehand_type:
            trigramtype = onehand_type
        else:
            raise RuntimeError("Onehand trigram type not recognized: " + trigram)
    elif trigram_main_type == TrigramMainType.UNTYPABLE:
        return "untypable"
    else:
        raise RuntimeError("Trigram type not recognized: " + trigram)
    return trigramtype


def get_onehand_trigram_type(
    trigram: str, mapping: EasyRollingTrigramsMap | None, hands: Hands
) -> OnehandTrigramType | None:
    indices, keytypes = hands.where(trigram)

    if len(set(keytypes)) != 1:
        # Not a onehand.
        return None
    if not len(indices) == 3:
        return None

    if mapping is not None:
        easy_type = mapping.get(typing.cast(tuple[int, int, int], tuple(indices)))
        if easy_type:
            return "rolling-easy"

    current_hand_type = keytypes[0]
    hand = {"Left": hands.left, "Right": hands.right}[current_hand_type]
    matrix_positions = hand.get_matrix_positions(indices)
    column_diffs = (
        matrix_positions[1][0] - matrix_positions[0][0],
        matrix_positions[2][0] - matrix_positions[1][0],
    )
    if (current_hand_type == "Left" and all(x < 0 for x in column_diffs)) or (
        current_hand_type == "Right" and all(x > 0 for x in column_diffs)
    ):
        return "rolling-out"
    if (current_hand_type == "Left" and all(x > 0 for x in column_diffs)) or (
        current_hand_type == "Right" and all(x < 0 for x in column_diffs)
    ):
        return "rolling-other"
    if column_diffs[0] * column_diffs[1] < 0:
        return "redir"
    if any(x == 0 for x in column_diffs):
        return "samecol"

    # Should never happen, though.
    raise RuntimeError("Trigram type not recognized: " + trigram)


def get_easy_rolling_type_mapping(
    easy_rolling_trigrams: dict[str, np.ndarray] | None, hands: Hands
) -> EasyRollingTrigramsMap | None:
    """Creates mapping from key sequences (tuple of indices) to easy rolling trigram
    types."""
    if easy_rolling_trigrams is None:
        return None

    conf_left = easy_rolling_trigrams

    shapes_left = get_easy_rolling_shapes(conf_left)

    # The configuration is created for left hand; need to vertically mirror for the
    # right hand to get the inward/outward directions correctly.
    conf_right = {k: np.flip(val, axis=1) for k, val in conf_left.items()}
    shapes_right = get_easy_rolling_shapes(conf_right)

    out = dict()
    # left and right hand might in theory have slightly different index-to-matrix_positions
    # mapping (in most cases, both hands would return same dict)
    for hand, shapes in zip((hands.left, hands.right), (shapes_left, shapes_right)):
        out.update(_get_easy_rolling_indices_to_names_map(hand, shapes))

    return out


def _get_easy_rolling_indices_to_names_map(
    hand: Hand, shapes: dict[TrigramShapeCoords, str]
) -> dict[tuple[int, int, int], str]:
    expected_fingers = [FingerType.R, FingerType.M, FingerType.I]
    mapping = {v: k for k, v in hand.matrix_positions.items()}
    out = dict()
    for starting_pos in mapping.keys():
        for mpdiffs, trigram_name in shapes.items():
            if not (indices := _get_indices(starting_pos, mpdiffs, mapping)):
                continue
            if hand.get_fingers(indices) != expected_fingers:
                continue
            out[indices] = trigram_name
    return out


def _get_indices(
    starting_pos: tuple[int, int],
    mpdiffs: TrigramShapeCoords,
    indices_mapping: dict[tuple[int, int], int],
) -> tuple[int, int, int] | None:
    typed_indices = []
    for mpdiff in mpdiffs:
        current_pos = (starting_pos[0] + mpdiff[0], starting_pos[1] + mpdiff[1])
        if current_pos not in indices_mapping:
            return None
        typed_indices.append(indices_mapping[current_pos])
    if not len(typed_indices) == 3:
        raise RuntimeError("len(typed_indices) != 3. This should never happen")

    return typing.cast(tuple[int, int, int], tuple(typed_indices))


def get_easy_rolling_shapes(
    easy_rolling_trigrams: dict[str, np.ndarray]
) -> dict[TrigramShapeCoords, str]:
    """Get a dictionary where keys are trigram shapes as tuples of matrix positions
    relative to the first key (matrix position diffs), and values are the trigram names.
    """
    out = dict()
    for trigram_name, shapearr in easy_rolling_trigrams.items():
        for shapetuple in _iter_shape_tuples(shapearr):
            out[shapetuple] = trigram_name
    return out


def _iter_shape_tuples(
    shapearr: np.ndarray,
) -> Iterable[tuple[Coord, Coord, Coord]]:
    """Iterates over a shape array and returns all possible shapes as tuples of matrix positions
    relative to the first key. For example:

    >>> shapearr = np.array(
            [
                [0, 3, 0],
                [0, 2, 3],
                [1, 0, 3],
            ]
        )
    >>> list(_iter_shape_tuples(shapearr))
        [
            ((0, 0), (1, -1), (1, -2)),
            ((0, 0), (1, -1), (2, -1)),
            ((0, 0), (1, -1), (2, 0))
        ]

    The first tuple of each shapetuple will always be (0, 0). The two values in each
    tuple are (column_diff, row_diff), and coordinate system is same as in matrix_positions;
    values grow from top to bottom and left to right.
    """

    locations_of_one = np.argwhere(shapearr == 1)
    locations_of_two = np.argwhere(shapearr == 2)
    locations_of_three = np.argwhere(shapearr == 3)

    if len(locations_of_one) != 1:
        raise ValueError("easy_rolling_trigrams must have exactly one 1")
    elif len(locations_of_two) != 1:
        raise ValueError("easy_rolling_trigrams must have exactly one 2")
    elif not len(locations_of_three) > 0:
        raise ValueError("easy_rolling_trigrams must have at least one 3")

    ref_location = tuple(int(x) for x in locations_of_one[0])

    def start_from_origin(x):
        # Note that argwhere returns (y, x) coordinates
        return (int(x[1] - ref_location[1]), int(x[0] - ref_location[0]))

    loc_of_one = (0, 0)
    loc_of_two = start_from_origin(locations_of_two[0])

    for loc in locations_of_three:
        yield loc_of_one, loc_of_two, start_from_origin(loc)
