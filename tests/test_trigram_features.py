from __future__ import annotations

import typing

import pytest

from granite_tools.app_types import TrigramMainType
from granite_tools.trigram_features import (
    EasyRollingFeatureFlag,
    RedirFeatureFlag,
    SingleFingerPatternFlag,
    TrigramFeatures,
    Vert2uFeatureFlag,
    _is_sfb,
    _is_sfs,
    _is_sfsb,
    _is_sft,
    _is_sftb,
    bigram_is_vert2u,
    get_redir_feature,
    get_single_finger_pattern,
    get_vert2u_feature,
)
from granite_tools.trigram_types import UnTypableTrigramError

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands


class TestTrigramFeatures:

    @pytest.mark.parametrize(
        "trigram, expected_args",
        [
            # fmt: off
            ("STB", ("onehand", "SFB", "v1x", None, None)),
            ("STT", ("onehand", None, None, None, None)),
            ("GTT", ("onehand", "SFB", None, None, None)),
            ("BTT", ("onehand", "SFB", "v1x", None, None)),
            ("BTB", ("onehand", "SFTb", "v2x", None, None)),
            ("RTB", ("onehand", "SFT", "v1x", None, None)),
            ("RTG", ("onehand", "SFT", None, None, None)),
            ("FAR", ("onehand", "SFS", None, None, None)),
            ("FGF", ("onehand", "SFTb", None, None, None)),
            ("MUM", ("onehand", "SFTb", "v2x", None, None)),
            ("MMM", ("onehand", None, None, None, None)),
            ("FDF", ("onehand", "SFSb", None, None, None)),
            ("WCW", ("onehand", "SFSb", "v2x", None, None)),
            ("ZTE", ("onehand", None, "v1x", "redir", None)),
            ("WCR", ("onehand", None, "v2x", None, None)),
            ("DFS", ("onehand", None, None, "redir", None)),
            ("WEQ", ("onehand", None, None, "redir", None)),
            ("AGD", ("onehand", None, None, "redir", None)),
            ("FGS", ("onehand", "SFB", None, "redir", None)),
            ("SDO", ("balanced", None, None, None, None)),
            ("SIO", ("balanced", None, None, None, None)),
            ("SOO", ("balanced", None, None, None, None)),
            ("OOS", ("balanced", None, None, None, None)),
            ("ODF", ("balanced", None, None, None, None)),
            ("ODD", ("balanced", None, None, None, None)),
            ("DOD", ("alternating", None, None, None, None)),
            ("MEM", ("alternating", None, None, None, None)),
            ("EME", ("alternating", None, None, None, None)),
            ("M M", ("alternating", None, None, None, None)),  # thumb, but still alternating
            ("XDF", ("onehand", None, None, None, "easy-rolling")),
            ("SDF", ("onehand", None, None, None, "easy-rolling")),
            ("XDV", ("onehand", None, None, None, "easy-rolling")),
            ("XCG", ("onehand", None, None, None, "easy-rolling")),
            ("SDT", ("onehand", None, None, None, "easy-rolling")),
            ("LKY", ("onehand", None, None, None, "easy-rolling")),
            ("LKJ", ("onehand", None, None, None, "easy-rolling")),
            ("JKL", ("onehand", None, None, None, None)),
            ("FDS", ("onehand", None, None, None, None)),
            # fmt: on
        ],
    )
    def test_multiple(
        self,
        trigram: str,
        expected_args: tuple[
            TrigramMainType,
            SingleFingerPatternFlag,
            Vert2uFeatureFlag,
            RedirFeatureFlag,
            EasyRollingFeatureFlag,
        ],
        hands_full: Hands,
    ) -> None:
        assert TrigramFeatures.from_string(trigram, hands_full) == TrigramFeatures(
            *expected_args
        )

    @pytest.mark.parametrize(
        "args, trigram_subtype",
        [
            # fmt: off
            (("onehand", "SFB", "v1x", None, None), "SFB|v1x"),
            (("onehand", "SFB", None, "redir", None), "SFB|redir"),
            (("onehand", "SFB", "v2x", "redir", None), "SFB|redir|v2x"),
            (("balanced", None, None, None, None), "balanced"),
            (("onehand", None, None, None, None), "onehand"),
            (("onehand", None, None, None, "easy-rolling"), "easy-rolling"),
            # hypothetical (easy-rolling) trigram with v1x flag
            (("onehand", None, "v1x", None, "easy-rolling"), "easy-rolling|v1x"),
            # A special case where the onehand prefix is preserved
            (("onehand", None, "v1x", None, None), "onehand|v1x"),
            # fmt: on
        ],
    )
    def test_trigram_subtype(
        self,
        args: tuple[
            TrigramMainType,
            SingleFingerPatternFlag,
            Vert2uFeatureFlag,
            RedirFeatureFlag,
            EasyRollingFeatureFlag,
        ],
        trigram_subtype: str,
    ) -> None:
        assert TrigramFeatures(*args).trigram_subtype == trigram_subtype

    def test_with_visualization_symbols(self, hands_full: Hands):
        # It's not possible to create a trigram with visualization symbols by default
        with pytest.raises(UnTypableTrigramError):
            features = TrigramFeatures.from_string("_+T", hands_full)

        # ..but if you tell it to use visualization symbols, it works
        features = TrigramFeatures.from_string(
            "_+T", hands_full, use_visualization_symbols=True
        )
        assert features == TrigramFeatures("onehand", "SFB", None, None, None)


class TestGetSingleFingerPattern:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", "SFB"),  # TB is typed with left index
            ("STT", None),  # TT is a repeat, not SFB
            ("GTT", "SFB"),  # TT is a repeat, but GT is a SFB
            ("UHO", "SFB"),  # UH is typed with right index
            ("STD", None),  # regular onehand trigram
            ("YHF", None),  # balanced trigram
            ("RTB", "SFT"),  # RTB is typed with left index
            ("FAR", "SFS"),  # F and R are typed with left index
            ("FGF", "SFTb"),  # typed with left index
            ("MUM", "SFTb"),
            ("FDF", "SFSb"),
            ("DQE", "SFS"),  # First and last key are typed with left middle finger
        ],
    )
    def test_multiple(
        self, trigram: str, expected: str | None, hands_full: Hands
    ) -> None:
        assert get_single_finger_pattern(trigram, hands_full) == expected


class TestIsSfb:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", True),  # TB is typed with left index
            ("STT", False),  # TT is a repeat, not SFB
            ("GTT", True),  # TT is a repeat, but GT is a SFB
            ("UHO", True),  # UH is typed with right index
            ("STD", False),  # regular onehand trigram
            ("RTB", False),  # RTB is typed with left index
            ("FAR", False),  # F and R are typed with left index
            ("FGF", False),
            ("FGD", True),
            ("FDF", False),  # SFSb
            ("FGT", False),  # SFT
        ],
    )
    def test_multiple(self, trigram: str, expected: bool, hands_full: Hands) -> None:

        indices, keytypes = hands_full.where(trigram)
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right
        fingers = [hand.get_finger(idx) for idx in indices]

        assert _is_sfb(indices, fingers) == expected


class TestIsSft:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", False),  # SFB
            ("GTT", False),  # SFB (all with same finger, though)
            ("RTB", True),  # SFT
            ("FDF", False),  # SFSb
            ("FGF", False),  # SFTb since first and last key the same
            ("FGT", True),  # SFT
        ],
    )
    def test_multiple(self, trigram: str, expected: bool, hands_full: Hands) -> None:

        indices, keytypes = hands_full.where(trigram)
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right
        fingers = [hand.get_finger(idx) for idx in indices]

        assert _is_sft(indices, fingers) == expected


class TestIsSftb:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", False),  # SFB
            ("GTT", False),  # SFB (all with same finger, though)
            ("RTB", False),  # SFT
            ("FGF", True),  # SFTb since first and last key the same
            ("MUM", True),  # SFTb since first and last key the same
            ("FGT", False),  # SFT
            ("FDF", False),  # SFSb
            ("MMM", False),  # 3-repeat
        ],
    )
    def test_multiple(self, trigram: str, expected: bool, hands_full: Hands) -> None:

        indices, keytypes = hands_full.where(trigram)
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right
        fingers = [hand.get_finger(idx) for idx in indices]

        assert _is_sftb(indices, fingers) == expected


class TestIsSfs:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", False),  #  SFB
            ("STT", False),
            ("GTT", False),  # SFB
            ("STD", False),  # regular onehand trigram
            ("RTB", False),  # SFT
            ("FAR", True),  # F and R are typed with left index
            ("FGD", False),  # SFB
            ("FGF", False),  # SFTb since first and last key the same
            ("FGT", False),  # SFT
            ("FDF", False),  # SFSb
            ("DQE", True),  # D and E are typed with left middle finger
        ],
    )
    def test_multiple(self, trigram: str, expected: bool, hands_full: Hands) -> None:

        indices, keytypes = hands_full.where(trigram)
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right
        fingers = [hand.get_finger(idx) for idx in indices]

        assert _is_sfs(indices, fingers) == expected


class TestIsSfsb:
    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("STB", False),  #  SFB
            ("STT", False),  # repeat
            ("GTT", False),  # SFB
            ("STD", False),  # regular onehand trigram
            ("RTB", False),  # SFT
            ("FAR", False),  # F and R are typed with left index
            ("FGD", False),  # SFB
            ("FGF", False),  # SFTb
            ("FDF", True),
            ("FGT", False),  # SFT
            ("DQE", False),  # D and E are typed with left middle finger
        ],
    )
    def test_multiple(self, trigram: str, expected: bool, hands_full: Hands) -> None:

        indices, keytypes = hands_full.where(trigram)
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right
        fingers = [hand.get_finger(idx) for idx in indices]

        assert _is_sfsb(indices, fingers) == expected


class TestGetRedirFeature:

    REDIR = "redir"

    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("SDF", None),
            ("SFD", REDIR),
            ("FSD", REDIR),
            ("DSF", REDIR),
            ("DFS", REDIR),
            ("WEQ", REDIR),
            ("L;K", REDIR),
            ("OPY", REDIR),
            ("OPU", REDIR),
            ("OPI", REDIR),
            ("DFD", None),  # same key twice
            ("DFE", None),  # SFS instead
            ("DFQ", REDIR),
            ("XFD", REDIR),
            ("EFX", REDIR),
            ("DGF", REDIR),  # redir AND SFB
        ],
    )
    def test_multiple(
        self, trigram: str, expected: str | None, hands_full: Hands
    ) -> None:
        assert get_redir_feature(trigram, hands_full) == expected


class TestGetVert2uFeature:

    def test_simple(self, hands_full: Hands) -> None:
        assert get_vert2u_feature("foo", hands_full) is None
        assert get_vert2u_feature("wcr", hands_full) == "v2x"
        assert get_vert2u_feature("ccr", hands_full) == "v1x"

    # another test with multiple trigrams. parametrize

    @pytest.mark.parametrize(
        "trigram, expected",
        [
            ("ZWV", "v1x"),
            ("U,O", "v2x"),
            ("YMI", "v1x"),
            (" R ", None),
            (" RB", "v1x"),  # can have 1x penalty even with thumb
            (" TV", "v1x"),  # can have 1x penalty even with thumb
            ("T V", None),  # both bigram have thumb
            ("XEV", None),  # XE and EV are exceptions (no penalty)
            ("MI.", None),  # MI and .I are exceptions (no penalty)
            ("QCF", "v1x"),
            ("QCP", None),  # Not a onehand -> never vert2u
            ("ZEC", "v1x"),  # ZE is exception (no penalty), EC is 2u
            ("PMY", "v2x"),
        ],
    )
    def test_multiple(
        self,
        trigram: str,
        expected: str,
        hands_full: Hands,
    ) -> None:
        assert get_vert2u_feature(trigram, hands_full) == expected


class TestBigramIsVert2u:

    @pytest.mark.parametrize(
        "bigram, expected",
        [
            ("QX", True),
            ("XQ", True),
            ("QV", True),
            ("ZW", True),
            ("U,", True),
            (",O", True),
            ("ZQ", True),  # pinky 2u SFB
            ("VR", True),  # index 2u SFB
            ("MU", True),  # index 2u SFB (right)
            ("CT", True),  # middle bottom, index (outer) top
            ("TC", True),  # middle bottom, index (outer) top
            ("CR", True),  # middle bottom, index (inner) top
            ("DV", False),  # 1u
            ("DD", False),  # same key
            ("DF", False),  # same row
            (" F", False),  # 2u but thumb
            ("F ", False),  # 2u but thumb
            (" Q", False),  # 3u but thumb
            (" T", False),  # 3u but thumb
            ("XE", False),  # From exception list r bottom <-> m top
            ("EX", False),  # From exception list r bottom <-> m top
            ("ZR", False),  # From exception list p bottom <-> i top
            ("ZT", True),  # Close to ZR but with inner index keyt top
            ("WV", False),  # From exception list i bottom <-> r top
        ],
    )
    def test_bigram_is_vert2u(
        self, bigram: str, expected: bool, hands_full: Hands
    ) -> None:

        indices, keytypes = hands_full.where(bigram)
        onehand = keytypes[0] == keytypes[1]
        if not onehand or keytypes[0] == "Untypable":
            raise Exception(f'The "{bigram}" is not a onehand bigram!')
        hand = hands_full.left if keytypes[0] == "Left" else hands_full.right

        assert (
            bigram_is_vert2u(indices, hand, hands_full.config.vert2u_penalties)
            == expected
        )
        # Also the reversed bigram must have same outcome
        assert (
            bigram_is_vert2u(indices[::-1], hand, hands_full.config.vert2u_penalties)
            == expected
        )
