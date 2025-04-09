from __future__ import annotations

import typing

import pytest

from granite_tools.trigram_types import UnTypableTrigramError, get_trigram_type

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands


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

BALANCED_TRIGRAMS = [
    "QWO",
    "CEO",
    "OMG",
    "LXS",
    "SLO",
    ";DF",
    "W,,",
    " PP",
    "PP ",
]

ALTERNATING_TRIGRAMS = [
    "LEL",
    "DOD",
    "I I",
    "O O",
    "SOX",
    "XOS",
    "LAP",
]

ONEHAND_TRIGRAMS = (
    OUTWARD_ROLLING_TRIGRAMS
    + OTHER_ROLLING_TRIGRAMS
    + REDIR_TRIGRAMS
    + SAMECOL_TRIGRAMS
)


class TestGetMainType:

    def test_onehand(self, hands_full: Hands):
        assert get_trigram_type("BAE", hands_full) == "onehand"

    def test_alternating(self, hands_full: Hands):
        assert get_trigram_type("SOF", hands_full) == "alternating"

    def test_balanced(self, hands_full: Hands):
        assert get_trigram_type("SFO", hands_full) == "balanced"

    def test_untypable(self, hands_full: Hands):
        with pytest.raises(UnTypableTrigramError):
            get_trigram_type("ÅÅÅ", hands_full)

    def test_non_trigram(self, hands_full: Hands):
        with pytest.raises(ValueError):
            get_trigram_type("XX", hands_full)

    EXAMPLES = list(
        zip(
            ONEHAND_TRIGRAMS + BALANCED_TRIGRAMS + ALTERNATING_TRIGRAMS,
            ["onehand"] * len(ONEHAND_TRIGRAMS)
            + ["balanced"] * len(BALANCED_TRIGRAMS)
            + ["alternating"] * len(ALTERNATING_TRIGRAMS),
        )
    )

    @pytest.mark.parametrize(
        "trigram, expected",
        EXAMPLES,
    )
    def test_multiple(
        self,
        trigram: str,
        expected: str,
        hands_full: Hands,
    ):
        assert get_trigram_type(trigram, hands_full) == expected
