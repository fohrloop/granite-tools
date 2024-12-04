from __future__ import annotations

import typing

from granite_tools.scorer.trigram_creation import (
    _get_reference_trigram,
    create_balanced_trigram,
    create_onehand_trigram,
    create_redir_trigram,
    create_skipgram_trigram,
)

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands


class TestCreateOnehandTrigram:

    def test_asd(self, hands_full: Hands):
        assert create_onehand_trigram(
            [8, 11, 14], rolldirection="in", side="Left", hands=hands_full
        ) == ("ASD", (14, 11, 8))
        assert create_onehand_trigram(
            [8, 11, 14], rolldirection="out", side="Left", hands=hands_full
        ) == ("DSA", (8, 11, 14))
        assert create_onehand_trigram(
            [8, 11, 14], rolldirection="in", side="Right", hands=hands_full
        ) == (";LK", (14, 11, 8))
        assert create_onehand_trigram(
            [8, 11, 14], rolldirection="out", side="Right", hands=hands_full
        ) == ("KL;", (8, 11, 14))

    def test_qcr(self, hands_full: Hands):
        indices = [13, 9, 4]
        assert create_onehand_trigram(
            indices, rolldirection="in", side="Left", hands=hands_full
        ) == ("QCR", (13, 9, 4))
        assert create_onehand_trigram(
            indices, rolldirection="out", side="Left", hands=hands_full
        ) == ("RCQ", (4, 9, 13))
        assert create_onehand_trigram(
            indices, rolldirection="in", side="Right", hands=hands_full
        ) == ("P,U", (13, 9, 4))
        assert create_onehand_trigram(
            indices, rolldirection="out", side="Right", hands=hands_full
        ) == ("U,P", (4, 9, 13))


class TestCreateRedirTrigram:

    def test_start_index_1(self):
        indices = [1, 2, 3]
        assert create_redir_trigram(indices, 1) == (2, 3, 1)

    def test_start_index_2(self):
        indices = [1, 2, 3]
        assert create_redir_trigram(indices, 2) == (3, 1, 2)


class TestCreateBalancedTrigram:

    def test_sdf(self, hands_full: Hands):
        # left, inwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [11, 8, 5]  # SDF on left
        assert create_balanced_trigram(indices, hand_left, hand_right, 0) == "LDF"
        assert create_balanced_trigram(indices, hand_left, hand_right, 2) == "SDJ"

    def test_fds(self, hands_full: Hands):
        # left, outwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [5, 8, 11]  # FDS on left
        assert create_balanced_trigram(indices, hand_left, hand_right, 0) == "JDS"
        assert create_balanced_trigram(indices, hand_left, hand_right, 2) == "FDL"

    def test_oiu(self, hands_full: Hands):
        # right, inwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [10, 7, 4]
        # OIU = 10, 7, 4 on right
        assert create_balanced_trigram(indices, hand_right, hand_left, 0) == "WIU"
        assert create_balanced_trigram(indices, hand_right, hand_left, 2) == "OIR"

    def test_uio(self, hands_full: Hands):
        # right, outwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [4, 7, 10]  # UIO
        assert create_balanced_trigram(indices, hand_right, hand_left, 0) == "RIO"
        assert create_balanced_trigram(indices, hand_right, hand_left, 2) == "UIW"


class TestCreateSkipgramTrigram:

    def test_sdf(self, hands_full: Hands):
        # left, inwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [11, 8, 5]  # SDF on left
        assert create_skipgram_trigram(indices, hand_left, hand_right, 0) == "DLF"
        assert create_skipgram_trigram(indices, hand_left, hand_right, 1) == "SKF"
        assert create_skipgram_trigram(indices, hand_left, hand_right, 2) == "SJD"

    def test_fds(self, hands_full: Hands):
        # left, inwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [5, 8, 11]  # FDS on left
        assert create_skipgram_trigram(indices, hand_left, hand_right, 0) == "DJS"
        assert create_skipgram_trigram(indices, hand_left, hand_right, 1) == "FKS"
        assert create_skipgram_trigram(indices, hand_left, hand_right, 2) == "FLD"

    def test_oiu(self, hands_full: Hands):
        # right, inwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [10, 7, 4]
        # OIU = 10, 7, 4 on right
        assert create_skipgram_trigram(indices, hand_right, hand_left, 0) == "IWU"
        assert create_skipgram_trigram(indices, hand_right, hand_left, 1) == "OEU"
        assert create_skipgram_trigram(indices, hand_right, hand_left, 2) == "ORI"

    def test_uio(self, hands_full: Hands):
        # right, outwards rolling
        hand_left = hands_full.left
        hand_right = hands_full.right
        indices = [4, 7, 10]  # UIO
        assert create_skipgram_trigram(indices, hand_right, hand_left, 0) == "IRO"
        assert create_skipgram_trigram(indices, hand_right, hand_left, 1) == "UEO"
        assert create_skipgram_trigram(indices, hand_right, hand_left, 2) == "UWI"


def test_get_reference_trigram():

    assert _get_reference_trigram("SDF") == ""
    assert _get_reference_trigram("[da") == ""
    assert _get_reference_trigram("da]") == ""
    assert _get_reference_trigram("[asd]") == "asd"
    assert _get_reference_trigram('["asd"]') == "asd"
    assert _get_reference_trigram("['asd']") == "asd"
    assert _get_reference_trigram('[as"]') == 'as"'
    assert _get_reference_trigram('["as]') == '"as'
