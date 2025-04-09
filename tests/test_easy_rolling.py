from __future__ import annotations

import typing

import numpy as np
import pytest

from granite_tools.easy_rolling import (
    _get_easy_rolling_shapes,
    _iter_shape_tuples,
    get_easy_rolling_type_mapping,
)
from granite_tools.hands import get_hands_data

if typing.TYPE_CHECKING:
    from granite_tools.config import Config
    from granite_tools.easy_rolling import EasyRollingTrigramsMap

EASY_ROLLING_TYPE_EXAMPLES = [
    ("scv", "snow-scoop"),  # left hand
    ("l,m", "snow-scoop"),  # right hand
    ("okj", "snow-scoop"),  # right hand, top row
    ("no;", None),
    ("ser", "shield"),
    ("set", "shield"),
    ("xdr", "shield"),
    ("xdg", "shield"),
    ("xdb", "shield"),
    ("xdt", None),
    ("wer", "hammer"),
    ("WEG", "hammer"),
    ("sdf", "hammer"),
    ("sdg", "hammer"),
    ("xcv", "hammer"),
    ("scf", None),
    ("asd", None),
    ("axc", None),  # snow-scoop but starts with pinky
]

EASY_ROLLING_TRIGRAMS = [x[0] for x in EASY_ROLLING_TYPE_EXAMPLES if x[1]]


@pytest.fixture
def mapping_full(config_full: Config) -> EasyRollingTrigramsMap:
    hands = get_hands_data(config_full)
    mapping = get_easy_rolling_type_mapping(config_full.easy_rolling_trigrams, hands)
    return mapping


class TestGetEasyRollingTypeMapping:

    @pytest.mark.parametrize(
        "trigram, expected",
        EASY_ROLLING_TYPE_EXAMPLES,
    )
    def test_get_easy_rolling_type_mapping(
        self, trigram, expected, config_full: Config
    ):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        indices = tuple(hands.where(trigram)[0])
        assert mapping.get(indices) == expected


class TestGetEasyRollingShapes:
    def test_get_easy_rolling_shapes(self, config_full: Config):
        shapes = _get_easy_rolling_shapes(config_full.easy_rolling_trigrams)

        assert shapes[((0, 0), (1, 1), (2, 1))] == "snow-scoop"
        assert shapes[((0, 0), (1, 0), (2, 0))] == "hammer"


class TestIterShapeTuples:

    def test_simple_diagonal(self):

        shapearr = np.array(
            [
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ]
        )

        assert list(_iter_shape_tuples(shapearr)) == [
            ((0, 0), (1, 1), (2, 2)),
        ]

    def test_multiple_threes_downwards(self):

        shapearr = np.array(
            [
                [1, 3, 0],
                [0, 2, 3],
                [0, 0, 3],
            ]
        )

        loc_first = (0, 0)
        loc_second = (1, 1)
        out = list(_iter_shape_tuples(shapearr))
        assert out == [
            (loc_first, loc_second, (1, 0)),
            (loc_first, loc_second, (2, 1)),
            (loc_first, loc_second, (2, 2)),
        ]

    def test_multiple_threes_upwards(self):

        shapearr = np.array(
            [
                [0, 3, 0],
                [0, 2, 3],
                [1, 0, 3],
            ]
        )

        loc_first = (0, 0)
        loc_second = (1, -1)
        out = list(_iter_shape_tuples(shapearr))
        assert out == [
            (loc_first, loc_second, (1, -2)),
            (loc_first, loc_second, (2, -1)),
            (loc_first, loc_second, (2, 0)),
        ]
