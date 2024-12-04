import numpy as np
import pytest

from granite_tools.config import Config
from granite_tools.hands import Hands, get_hands_data
from granite_tools.scorer.trigramtype import (
    _iter_shape_tuples,
    get_easy_rolling_shapes,
    get_easy_rolling_type_mapping,
    get_onehand_trigram_type,
)

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

OUTWARD_ROLLING_TRIGRAMS = [
    "dsa",
    "gfd",
    "grd",
    "be1",
    "xz1",
    "jil",
    "m,.",
    "u.2",
]

OTHER_ROLLING_TRIGRAMS = [
    "qwe",  # pinky
    "zse",  # pinky, diagornal
    "wdv",  # ring, diagonal downwards
    "efg",
    "poi",
    ",jh",
    "kjh",
    "wcr",
]

REDIR_TRIGRAMS = [
    "xfd",
    "vgr",
    "jlk",
    "klj",
    ";hj",
    "sfd",
    "crd",
    "wfx",
]

SAMECOL_TRIGRAMS = [
    "qaz",
    "sdd",
    "sdc",
    "fff",
    "fft",
    "kjm",
    "lp;",
    ",ju",
]


class TestGetRollingType:

    @pytest.mark.parametrize(
        "trigram",
        EASY_ROLLING_TRIGRAMS,
    )
    def test_easy_rolling(self, trigram, config_full: Config):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        assert get_onehand_trigram_type(trigram, mapping, hands) == "rolling-easy"

    @pytest.mark.parametrize(
        "trigram",
        OUTWARD_ROLLING_TRIGRAMS,
    )
    def test_rolling_out(self, trigram, config_full: Config):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        assert get_onehand_trigram_type(trigram, mapping, hands) == "rolling-out"

    @pytest.mark.parametrize(
        "trigram",
        OTHER_ROLLING_TRIGRAMS,
    )
    def test_rolling_other(self, trigram, config_full: Config):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        assert get_onehand_trigram_type(trigram, mapping, hands) == "rolling-other"

    @pytest.mark.parametrize(
        "trigram",
        REDIR_TRIGRAMS,
    )
    def test_redir(self, trigram, config_full: Config):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        assert get_onehand_trigram_type(trigram, mapping, hands) == "redir"

    @pytest.mark.parametrize(
        "trigram",
        SAMECOL_TRIGRAMS,
    )
    def test_samecol(self, trigram, config_full: Config):
        hands = get_hands_data(config_full)
        mapping = get_easy_rolling_type_mapping(
            config_full.easy_rolling_trigrams, hands
        )
        assert get_onehand_trigram_type(trigram, mapping, hands) == "samecol"


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


class TestEasyRollingTrigramBlueprints:
    def test_get_easy_rolling_trigram_blueprints(self, config_full: Config):
        shapes = get_easy_rolling_shapes(config_full.easy_rolling_trigrams)

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
