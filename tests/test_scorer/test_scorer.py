from __future__ import annotations

import typing
from textwrap import dedent
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from granite_tools.config import text_to_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreDict,
    TrigramScoreSets,
    _get_multiplier_from_cosine,
    average_abs_error,
    get_angle_between_two_vectors,
    get_bigram_vectors,
    get_cosine_of_trigram_angle,
    get_dirchange_multiplier,
    get_onehand_base_score,
    get_scaled_bigram_scores,
    get_score,
    get_sfb_in_onehand_multiplier,
    get_trigram_params_error_fun,
    get_vert2u_multiplier,
    group_trigram_scores,
    indices_to_family_name,
    load_trigram_scores,
    to_key_sequences,
    to_skipgram_key_sequences,
)

if typing.TYPE_CHECKING:
    from granite_tools.config import Config
    from granite_tools.hands import Hands


@pytest.fixture
def model_params() -> TrigramModelParameters:
    return TrigramModelParameters()


class TestGetVert2UMultiplier:

    coeff = 3.0

    def test_same_row(self, hands_full: Hands):
        indices = [10, 7, 4]
        assert get_vert2u_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_falses(self, hands_full: Hands):
        indices_list = [
            [10, 9, 5],
            [11, 9, 4],
            [9, 5, 1],
            [11, 7, 6],
            [3, 2, 6],
            [13, 12, 14],
        ]
        for indices in indices_list:
            assert get_vert2u_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_falses_middle_finger_top(self, hands_full: Hands):
        indices_list = [
            [12, 7, 9],  # NOT vert2u, since middle finger is used for the top row!
            [12, 7, 6],  # # NOT vert2u, since middle finger is used for the top row!
        ]
        for indices in indices_list:
            assert get_vert2u_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_2u_2u(self, hands_full: Hands):
        indices_list = [
            [13, 12, 7],
            [10, 9, 1],  # middle but not top!
            [13, 12, 13],
            [15, 10, 9],
            [9, 4, 3],
            [9, 4, 6],
        ]
        for indices in indices_list:
            assert (
                get_vert2u_multiplier(indices, hands_full.left, self.coeff)
                == self.coeff
            ), f"Failed for {indices}"

    def test_thumb_keys(self, hands_full: Hands):
        # If thumb key is present in the trigram, the multiplier should be 1.0
        indices_list = [
            [7, 18, 4],  # 3u jump with thumb
            [8, 18, 2],  # 2u jump with thumb
        ]
        for indices in indices_list:
            assert (
                get_vert2u_multiplier(indices, hands_full.left, self.coeff) == 1.0
            ), f"Failed for {indices}"


class TestGetSfbInOnehandMultiplier:

    coeff = 3.0

    def test_not_sfb(self, hands_full: Hands):
        indices_list = [
            [13, 12, 7],
            [9, 4, 10],
            [14, 11, 1],
            [16, 12, 7],
            [4, 8, 5],  # first and last same but not sfb!
        ]
        for indices in indices_list:
            assert (
                get_sfb_in_onehand_multiplier(indices, hands_full.left, self.coeff)
                == 1.0
            ), f"Failed for {indices}"

    def test_sft(self, hands_full: Hands):
        indices_list = [
            [4, 5, 6],  # same column index sft (should not get sfb score!)
            [16, 15, 13],  # pinky, not same column sft
        ]
        for indices in indices_list:

            assert (
                get_sfb_in_onehand_multiplier(indices, hands_full.left, self.coeff)
                == 1.0
            ), f"Failed for {indices}"

    def test_sfb(self, hands_full: Hands):
        indices_list = [
            [9, 4, 6],  # same column index
            [9, 4, 3],  # different column index
            [16, 15, 12],  # different column pinky
            [9, 14, 15],  # same column pinky
            [4, 5, 8],  # first and second
            [8, 4, 5],  # second and last
        ]
        for indices in indices_list:
            assert (
                get_sfb_in_onehand_multiplier(indices, hands_full.left, self.coeff)
                == self.coeff
            ), f"Failed for {indices}"


class TestGetDirChangeMultiplier:

    coeff = 3.5

    def test_no_change(self, hands_full: Hands):
        indices = [14, 11, 8]
        assert get_dirchange_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_90_deg(self, hands_full: Hands):
        indices = [11, 7, 5]
        assert get_dirchange_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_135_deg(self, hands_full: Hands):
        indices = [14, 8, 10]
        assert get_dirchange_multiplier(indices, hands_full.left, self.coeff) == 1.0

    def test_180_deg(self, hands_full: Hands):
        indices_lst = [
            [14, 8, 11],
            [9, 12, 6],
            [9, 15, 3],
        ]
        for indices in indices_lst:
            assert (
                get_dirchange_multiplier(indices, hands_full.left, self.coeff)
                == self.coeff
            ), f"Failed for {indices}"

    def test_180_but_with_repeat(self, hands_full: Hands):
        # If first and last fingers are same, the multiplier should be 1.0
        indices_lst = [
            [9, 12, 9],
            [8, 5, 8],
            [2, 5, 2],
        ]
        for indices in indices_lst:
            assert (
                get_dirchange_multiplier(indices, hands_full.left, self.coeff) == 1.0
            ), f"Failed for {indices}"

    def test_two_last_chars_same(self, hands_full: Hands):
        indices = [14, 8, 8]
        assert get_dirchange_multiplier(indices, hands_full.left, self.coeff) == 1.0

    # test config with thumb keys in place where there could be 180 degree redirect
    thumb_test_config = dedent(
        """
    key_indices:
    - [       0,  1,  2,  3,  4,     4,  3,  2,  1,  0       ]
    - [       5,  6,  7,  8,  9,     9,  8,  7,  6,  5,      ]

    hands:
    - [      Left, Left, Left, Left, Left,                Right, Right, Right, Right, Right       ]
    - [      Left, Left, Left, Left, Left,                Right, Right, Right, Right, Right       ]

    symbols_visualization:
    - [       1,  2,  3,  4,  5,     E,  D,  C,  B,  A       ]
    - [       6,  7,  8,  9,  0,     J,  I,  H,  G,  F,      ]

    finger_matrix:
    - [       p,  r,  m,  i,  i,      i,  i,  m,  r,  p       ]
    - [       p,  r,  m,  i,  t,      t,  i,  m,  r,  p       ]

    matrix_positions:
    - [       [1,0], [2,0], [3,0], [4,0], [5,0],                  [10,0], [11,0], [12,0], [13,0], [14,0]        ]
    - [       [1,2], [2,2], [3,2], [4,2], [5,2],                  [10,2], [11,2], [12,2], [13,2], [14,2]        ]
    """.strip(
            "\n"
        )
    )

    def test_with_thumb_keys(self):
        # If trigram contains a thumb key, the dirchange multiplier is 1.0
        config = text_to_config(self.thumb_test_config)
        hands = get_hands_data(config)
        indices_lst = [
            [7, 9, 8],  # 9 is a thumb key on same row.
            [8, 9, 7],
        ]
        for indices in indices_lst:
            assert (
                get_dirchange_multiplier(indices, hands.left, self.coeff) == 1.0
            ), f"Failed for {indices}"


class TestGetBigramVectors:

    def test_no_invert_y(self):

        assert get_bigram_vectors([(0, 0), (2, 1), (3, 1)], invert_y=False) == (
            (2, 1),
            (1, 0),
        )
        assert get_bigram_vectors([(0, 0), (-1, -5), (0, 2)], invert_y=False) == (
            (-1, -5),
            (1, 7),
        )
        assert get_bigram_vectors([(3, 2), (1, 1), (0, 0)], invert_y=False) == (
            (-2, -1),
            (-1, -1),
        )

    def test_invert_y(self):

        assert get_bigram_vectors([(0, 0), (2, 1), (3, 1)]) == ((2, -1), (1, 0))
        assert get_bigram_vectors([(0, 0), (-1, -5), (0, 2)]) == ((-1, 5), (1, -7))
        assert get_bigram_vectors([(3, 2), (1, 1), (0, 0)]) == ((-2, 1), (-1, 1))


class TestCosineOfTrigramAngle:
    """Reference for used matrix positions:

    - [       [1,0], [2,0], [3,0], [4,0], [5,0],                  [10,0], [11,0], [12,0], [13,0], [14,0]        ]
    - [[0,1], [1,1], [2,1], [3,1], [4,1], [5,1],                  [10,1], [11,1], [12,1], [13,1], [14,1], [15,1]]
    - [       [1,2], [2,2], [3,2], [4,2], [5,2],                  [10,2], [11,2], [12,2], [13,2], [14,2]        ]

    """

    def test_90_deg(self):
        for matrix_positions in [
            [(2, 1), (3, 1), (3, 0)],
            [(2, 1), (3, 1), (3, 2)],
            [(2, 1), (2, 2), (3, 2)],
            [(2, 1), (2, 2), (1, 2)],
            [(2, 1), (1, 1), (1, 2)],
            [(2, 1), (1, 1), (1, 0)],
            [(2, 1), (2, 0), (1, 0)],
            [(2, 1), (2, 0), (3, 0)],
        ]:
            cosine = get_cosine_of_trigram_angle(matrix_positions)
            assert cosine == pytest.approx(0)
            assert np.rad2deg(np.arccos(cosine)) == pytest.approx(90)

    def test_90_deg_more(self):
        for matrix_positions in [
            [(1, 1), (2, 0), (3, 1)],
            [(4, 1), (3, 2), (2, 1)],
        ]:
            cosine = get_cosine_of_trigram_angle(matrix_positions)
            assert cosine == pytest.approx(0)
            assert np.rad2deg(np.arccos(cosine)) == pytest.approx(90)

    def test_180_deg(self):
        for matrix_positions in [
            [(4, 1), (2, 1), (3, 1)],
            [(2, 0), (4, 0), (3, 0)],
        ]:
            cosine = get_cosine_of_trigram_angle(matrix_positions)
            assert cosine == pytest.approx(-1)
            assert np.rad2deg(np.arccos(cosine)) == pytest.approx(180)

    def test_127_deg(self):
        for matrix_positions in [
            [(1, 0), (2, 2), (3, 0)],  # QWERTY QXE
        ]:
            cosine = get_cosine_of_trigram_angle(matrix_positions)
            assert cosine == pytest.approx(to_cosine(126.8698976458))

    def test_127_deg(self):
        for matrix_positions in [
            [(1, 0), (3, 0), (2, 2)],  # QWERTY QEX
        ]:
            cosine = get_cosine_of_trigram_angle(matrix_positions)
            assert cosine == pytest.approx(to_cosine(116.56505))


class TestGetAngleBetween:

    def test_45_deg(self):
        u = (1, 1)
        v = (0, 2)
        assert get_angle_between_two_vectors(u, v) == pytest.approx(45.0)

    def test_90_deg(self):
        u = (-1, 0)
        assert get_angle_between_two_vectors(u, (0, 1)) == pytest.approx(90.0)
        assert get_angle_between_two_vectors(u, (0, -2)) == pytest.approx(90.0)
        assert get_angle_between_two_vectors(u, (0, -1.4)) == pytest.approx(90.0)

    def test_180_deg(self):
        u = (1, 0)
        assert get_angle_between_two_vectors(u, (-1, 0)) == pytest.approx(180.0)
        assert get_angle_between_two_vectors(u, (-2, 0)) == pytest.approx(180.0)

    def test_174_deg(self):
        u = (1, 0)
        theta = 174.28940
        assert get_angle_between_two_vectors(u, (-1, 0.1)) == pytest.approx(theta)
        assert get_angle_between_two_vectors(u, (-1, -0.1)) == pytest.approx(theta)


def to_cosine(angle):
    return np.cos(np.deg2rad(angle))


class TestMultiplierFromCosine:

    coeff = 3.0

    def test_close_to_90_deg(self):
        assert _get_multiplier_from_cosine(to_cosine(89.999), self.coeff) == 1.0

    def test_close_to_max(self):
        assert _get_multiplier_from_cosine(to_cosine(180), self.coeff) == self.coeff
        assert _get_multiplier_from_cosine(to_cosine(179.99), self.coeff) == self.coeff
        assert _get_multiplier_from_cosine(to_cosine(180.01), self.coeff) == self.coeff

    def test_halfway(self):

        cos_halfway = to_cosine(90 + (180 - 90) / 2)
        assert _get_multiplier_from_cosine(cos_halfway, self.coeff) == 1.0

    def test_small_angle(self):
        assert _get_multiplier_from_cosine(to_cosine(85), self.coeff) == 1.0
        assert _get_multiplier_from_cosine(to_cosine(45), self.coeff) == 1.0
        assert _get_multiplier_from_cosine(to_cosine(5), self.coeff) == 1.0
        assert _get_multiplier_from_cosine(to_cosine(0), self.coeff) == 1.0
        assert _get_multiplier_from_cosine(to_cosine(-85), self.coeff) == 1.0


class TestGetScore:

    def test_easy_rolling_onehand(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sdf"
        bigram1 = 1.6
        bigram2 = 1.4
        bigram_scores = {
            (11, 8): bigram1,  # SD
            (8, 5): bigram2,  # DF
        }

        params = model_params.model_copy()
        params.easy_rolling_coeff = 1.234

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_base_score = (3 / 4) * (bigram1 + bigram2)

        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["onehand_extra"] == 1.0  # nothing
        assert scoredct["rolling"] == params.easy_rolling_coeff
        assert scoredct["score"] == params.easy_rolling_coeff * expected_base_score

    def test_outward_rolling_onehand_with_sfb(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "hmp"
        bigram1 = 1.6
        bigram2 = 1.4
        bigram_scores = {
            (2, 6): bigram1,  # HM
            (6, 13): bigram2,  # MP
        }

        params = model_params.model_copy()
        params.sfb_in_onehand_coeff = 1.06

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_base_score = (3 / 4) * (bigram1 + bigram2)
        expected_multiplier = params.sfb_in_onehand_coeff
        onehand_base_score = expected_multiplier * expected_base_score
        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["onehand_extra"] == expected_multiplier
        assert scoredct["onehand_base_score"] == onehand_base_score
        assert scoredct["rolling"] == 1.0
        assert scoredct["score"] == onehand_base_score

    def test_onehand_samecol_with_sft(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "tfv"
        bigram1 = 1.34
        bigram2 = 1.56
        bigram_scores = {
            (1, 5): bigram1,  # TF
            (5, 6): bigram2,  # FV
        }

        params = model_params.model_copy()
        params.sft_coeff = 1.4

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_base_score = (3 / 4) * (bigram1 + bigram2)

        expected_multiplier = params.sft_coeff
        onehand_base_score = expected_multiplier * expected_base_score

        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["onehand_extra"] == expected_multiplier
        assert scoredct["onehand_base_score"] == onehand_base_score
        assert scoredct["rolling"] == 1.0  # non-rolling / samecol
        assert scoredct["score"] == onehand_base_score
        assert scoredct["trigramtype"] == "samecol"

    def test_onehand_with_180_deg(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "dfs"
        bigram_scores = {
            (8, 5): 1.4,  # DF
            (5, 11): 2.0,  # FS
        }
        params = model_params.model_copy()
        params.dirchange_coeff = 3.0

        base_score = (3 / 4) * (1.4 + 2.0)
        expected_score = params.dirchange_coeff * base_score

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        assert scoredct["ngram1_score"] == 1.4
        assert scoredct["ngram2_score"] == 2.0
        assert scoredct["base_score"] == base_score
        assert scoredct["dirchange"] == params.dirchange_coeff
        assert scoredct["score"] == expected_score

    def test_onehand_with_redir_on_ring_finger(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "dwr"
        bigram_scores = {
            (8, 10): 1.4,  # DW
            (10, 4): 2.0,  # WR
        }
        params = model_params.model_copy()
        params.dirchange_coeff = 100.0

        base_score = (3 / 4) * (1.4 + 2.0)
        expected_score = base_score

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        assert scoredct["ngram1_score"] == 1.4
        assert scoredct["ngram2_score"] == 2.0
        assert scoredct["base_score"] == base_score
        assert scoredct["dirchange"] == 1.0  # no 180 deg change
        assert scoredct["score"] == expected_score
        assert scoredct["trigramtype"] == "redir"

    def test_onehand_with_vert2u(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "wcr"
        bigram_scores = {
            (10, 9): 1.4,  # WC
            (9, 4): 2.0,  # CR
        }
        params = model_params.model_copy()
        params.dirchange_coeff = 2.0
        params.vert2u_coeff = 3.0
        base_score = (3 / 4) * (1.4 + 2.0)

        expected_score = params.vert2u_coeff * base_score

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        assert scoredct["ngram1_score"] == 1.4
        assert scoredct["ngram2_score"] == 2.0
        assert scoredct["base_score"] == base_score
        assert scoredct["vert2u"] == params.vert2u_coeff
        assert scoredct["score"] == pytest.approx(expected_score)

    def test_balanced_llr(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sdu"
        bigram_scores = {
            (11, 8): 1.6,  # SD
            (4,): 1.1,  # U
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 3.5

        expected_score = (
            params.balanced_b_coeff * bigram_scores[(11, 8)]
            + params.unigram_coeff * bigram_scores[(4,)]
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score

    def test_balanced_lrr(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sku"
        bigram_scores = {
            (11,): 1.6,  # S
            (8, 4): 1.1,  # KU
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 3.5

        expected_score = (
            params.balanced_b_coeff * bigram_scores[(8, 4)]
            + params.unigram_coeff * bigram_scores[(11,)]
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score

    def test_balanced_rrl(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "lkr"
        bigram_scores = {
            (11, 8): 1.4,  # LK
            (4,): 1.2,  # R
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 3.5

        expected_score = (
            params.balanced_b_coeff * bigram_scores[(11, 8)]
            + params.unigram_coeff * bigram_scores[(4,)]
        )

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score

    def test_balanced_rll(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "ldr"
        bigram_scores = {
            (11,): 1.6,  # L
            (8, 4): 1.1,  # DR
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 3.5

        expected_score = (
            params.balanced_b_coeff * bigram_scores[(8, 4)]
            + params.unigram_coeff * bigram_scores[(11,)]
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score

    def test_balanced_with_space(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "@dr"  # _ is space in the config
        bigram_scores = {
            (17,): 1.02,  # @ (inner thumb key on right
            (8, 4): 1.1,  # DR
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 2.0

        expected_score = (
            params.balanced_b_coeff * bigram_scores[(8, 4)]
            + params.unigram_coeff * bigram_scores[(17,)]
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score

    def test_balanced_left_left_untypable(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "sd("  # ( is not in the layout
        bigram_scores = {
            (11, 8): 1.6,  # SD
        }

        params = model_params.model_copy()
        params.balanced_b_coeff = 1.4

        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == 0

    def test_skipgram(self, model_params: TrigramModelParameters, hands_full: Hands):
        ngram = "skf"
        bigram_scores = {
            (11, 5): 1.2,  # SF
            (8,): 1.0,  # K
        }
        params = model_params.model_copy()
        params.unigram_coeff = 1.9
        params.skipgram_b_coeff = 1.045

        expected_score = (
            params.skipgram_b_coeff * bigram_scores[(11, 5)]
            + bigram_scores[(8,)] * params.unigram_coeff
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == bigram_scores[(11, 5)] + bigram_scores[(8,)]
        assert scoredct["unigram_coeff"] == params.unigram_coeff
        assert scoredct["skipgram_b_coeff"] == params.skipgram_b_coeff

    def test_skipgram_with_thumb(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "s@f"
        bigram_scores = {
            (11, 5): 1.2,  # SF
            (17,): 4.0,  # @ (inner thumb key on right)
        }
        params = model_params.model_copy()
        params.unigram_coeff = 1.9
        params.skipgram_b_coeff = 1.045

        expected_score = (
            params.skipgram_b_coeff * bigram_scores[(11, 5)]
            + bigram_scores[(17,)] * params.unigram_coeff
        )
        scoredct = get_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == bigram_scores[(11, 5)] + bigram_scores[(17,)]
        assert scoredct["unigram_coeff"] == params.unigram_coeff
        assert scoredct["skipgram_b_coeff"] == params.skipgram_b_coeff


class TestGetScaledBigramScores:

    def test_simple(self):
        bigram_scores = {
            (1, 2): 1.0,
            (2, 3): 3.0,  # halfway from 1 to 5
            (3, 4): 2.0,  # 1/4 from 1 to 4
            (4, 5): 4.0,  # 3/4 from 1 to 4
            (5, 6): 5.0,  # max
        }
        scaled_scores = get_scaled_bigram_scores(bigram_scores, newmax=10.0)
        expected_scores = {
            (1, 2): 1.0,
            (2, 3): 5.5,  # halfway from 1 to 10
            (3, 4): 1 + 4.5 / 2,  # 1/4 from 1 to 10
            (4, 5): 1 + 4.5 * 1.5,  # 3/4 from 1 to 10
            (5, 6): 10.0,  # max
        }
        assert scaled_scores == expected_scores

        exponent = 2.52
        scaled_scores = get_scaled_bigram_scores(
            bigram_scores, newmax=10.0, exponent=exponent
        )
        expected_scores = {k: v**exponent for k, v in expected_scores.items()}
        assert scaled_scores == expected_scores


class TestToKeySequences:

    def test_llr(self):
        assert to_key_sequences([1, 2, 3], ["Left", "Left", "Right"]) == (
            ((1, 2), (3,)),
            ("Left", "Right"),
        )

    def test_rll(self):
        assert to_key_sequences([4, 2, 3], ["Right", "Left", "Left"]) == (
            ((4,), (2, 3)),
            ("Right", "Left"),
        )


class TestToSkipgramKeySequences:
    def test_lrl(self):
        assert to_skipgram_key_sequences([1, 2, 3], ["Left", "Right", "Left"]) == (
            ((1, 3), (2,)),
            ("Left", "Right"),
        )

    def test_rlr(self):
        assert to_skipgram_key_sequences([1, 2, 3], ["Right", "Left", "Right"]) == (
            ((1, 3), (2,)),
            ("Right", "Left"),
        )


TESTDATA = b"""
[ABC]
DEF=2.0  
GHI =3.0 
JKL= 4.0

"""


class TestLoadTrigramScores:

    @patch("builtins.open", mock_open(read_data=TESTDATA))
    def test_simple(self):

        scores = load_trigram_scores("_nonexisting_file_")
        assert scores == {
            "ABC": {
                "DEF": 2.0,
                "GHI": 3.0,
                "JKL": 4.0,
            }
        }


class TestTrigramScoreSets:

    def test_simple(self, hands_full: Hands):
        dct = {
            "SEF": {
                "EFL": 2.0,
                "ELF": 3.0,
                "ESF": 4.0,
            },
            ".KY": {
                "XKY": 2.0,
                ".TK": 3.0,
                "Y.K": 4.0,
            },
            # SEF trigram family. Starts third set.
            "JIL": {
                "LiJ": 1.8,
            },
        }
        trigram_sets = TrigramScoreSets.from_trigram_scores(dct, hands_full)
        assert trigram_sets["SEF"].scores == {
            # The reference SEF is not included
            "EFL": 2.0,
            "ELF": 3.0,
            "ESF": 4.0,
        }
        assert trigram_sets["SEF"].ref_family == "SEF"

        assert trigram_sets[".KY"].scores == {
            "XKY": 2.0,
            ".TK": 3.0,
            "Y.K": 4.0,
        }
        assert trigram_sets[".KY"].ref_family == "XDT"

        assert trigram_sets["JIL"].scores == {
            "LIJ": 1.8,  # converted to upper case. Reference: JIL.
        }
        assert trigram_sets["JIL"].ref_family == "SEF"


class TestGetTrigramFamilyName:

    def test_left_inroll(self, hands_full: Hands):
        assert indices_to_family_name((5, 7, 11), hands_full) == "SEF"

    def test_left_outroll(self, hands_full: Hands):
        assert indices_to_family_name((11, 7, 5), hands_full) == "SEF"

    def test_mixed_order(self, hands_full: Hands):
        assert indices_to_family_name((7, 5, 11), hands_full) == "SEF"

    def test_right_outroll(self, hands_full: Hands):
        assert indices_to_family_name((1, 8, 12), hands_full) == "XDT"

    def test_right_inroll(self, hands_full: Hands):
        assert indices_to_family_name((12, 8, 1), hands_full) == "XDT"


class TestTrigramModelParameters:

    def test_as_tuple(self):

        params = TrigramModelParameters(
            vert2u_coeff=1,
            dirchange_coeff=2,
            balanced_b_coeff=3,
            unigram_coeff=4,
            skipgram_b_coeff=5,
            easy_rolling_coeff=6,
            bigram_raw_range_max=7,
            bigram_scaling_exponent=8,
            sft_coeff=9,
            sfb_in_onehand_coeff=10,
        )
        assert params.as_tuple() == tuple(range(1, 9))
        assert params.as_tuple(only_model_params=False) == tuple(range(1, 11))

    def test_from_config(self, config_full: Config):
        config = config_full.model_copy()
        config.vert2u_coeff = 1
        config.dirchange_coeff = 2
        config.balanced_b_coeff = 3
        config.unigram_coeff = 4
        config.skipgram_b_coeff = 5
        config.easy_rolling_coeff = 6
        config.bigram_raw_range_max = 7
        config.bigram_scaling_exponent = 8
        config.sft_coeff = 9
        config.sfb_in_onehand_coeff = 10
        params = TrigramModelParameters.from_config(config)
        assert params.as_tuple(only_model_params=False) == tuple(range(1, 11))


class TestTrigramParamsErrorFunction:

    def test_simple(self, model_params: TrigramModelParameters, hands_full: Hands):

        trigram_score_dct = {
            "SEF": {
                "ELF": 1.5,
            },
        }

        scoresets = TrigramScoreSets.from_trigram_scores(trigram_score_dct, hands_full)
        bigram_scores = {
            (11, 7): 1.6,  # SE
            (7, 5): 1.4,  # EF
            (11,): 1.15,  # S, L
        }

        # Form expected score
        score_sef = get_score("SEF", hands_full, model_params, bigram_scores)["score"]
        score_elf = get_score("ELF", hands_full, model_params, bigram_scores)["score"]
        estimated_score = score_elf / score_sef
        target_score = trigram_score_dct["SEF"]["ELF"]
        expected_score = ((estimated_score - target_score) ** 2) ** 0.5

        scorefunc = get_trigram_params_error_fun(scoresets, hands_full, bigram_scores)
        params_tuple = model_params.as_tuple()
        assert scorefunc(params_tuple) == expected_score

    def test_two_trigram_sets(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        trigram_score_dct = {
            "SEF": {
                "ELF": 3.0,
                "EFL": 2.5,
            },
            ".KY": {  # XDT trigram family
                "XKY": 2.0,
            },
        }

        scoresets = TrigramScoreSets.from_trigram_scores(trigram_score_dct, hands_full)

        bigram_scores = {
            (11, 7): 1.6,  # SE
            (7, 5): 1.4,  # EF
            (11,): 1.15,  # S, L
            (12, 8): 1.8,  # XD, .K
            (8, 1): 1.2,  # DT, KY
            (12,): 1.1,  # X, .
        }

        # Form expected score
        score_sef = get_score("SEF", hands_full, model_params, bigram_scores)["score"]
        score_elf = get_score("ELF", hands_full, model_params, bigram_scores)["score"]
        score_efl = get_score("EFL", hands_full, model_params, bigram_scores)["score"]
        score_dotky = get_score(".KY", hands_full, model_params, bigram_scores)["score"]
        score_xky = get_score("XKY", hands_full, model_params, bigram_scores)["score"]

        estimated_score_elf = score_elf / score_sef
        target_score_elf = trigram_score_dct["SEF"]["ELF"]
        estimated_score_efl = score_efl / score_sef
        target_score_efl = trigram_score_dct["SEF"]["EFL"]
        estimated_score_xky = score_xky / score_dotky
        target_score_xky = trigram_score_dct[".KY"]["XKY"]

        expected_score_squared = (
            (estimated_score_elf - target_score_elf) ** 2
            + (estimated_score_efl - target_score_efl) ** 2
            + (estimated_score_xky - target_score_xky) ** 2
        ) / 3
        expected_score = expected_score_squared**0.5
        scorefunc = get_trigram_params_error_fun(scoresets, hands_full, bigram_scores)
        params_tuple = model_params.as_tuple()
        assert scorefunc(params_tuple) == expected_score


class TestGroupTrigramScores:

    qxd1 = {
        "trigram_family_name": "QXD",
        "trigram": "D.P",
        "reference_trigram": "K.P",
        "scaled_target_score": 0.82,
        "scaled_estimated_score": 0.4,
        "estimated_score": 2.2,
    }
    qxd2 = {
        "trigram_family_name": "QXD",
        "trigram": ".DP",
        "reference_trigram": "K.P",
        "scaled_target_score": 0.37,
        "scaled_estimated_score": 0.5,
        "estimated_score": 2.63,
    }
    sdf1 = {
        "trigram_family_name": "SDF",
        "trigram": "SJD",
        "reference_trigram": "SDF",
        "scaled_target_score": 1.2,
        "scaled_estimated_score": 0.56,
        "estimated_score": 1.2,
    }
    sdf2 = {
        "trigram_family_name": "SDF",
        "trigram": "DFS",
        "reference_trigram": "SDF",
        "scaled_target_score": 2.0,
        "scaled_estimated_score": 1.0,
        "estimated_score": 2.1,
    }
    wev1 = {
        "trigram_family_name": "WEV",
        "trigram": "WEM",
        "reference_trigram": "WEV",
        "scaled_target_score": 0.6,
        "scaled_estimated_score": 0.4,
        "estimated_score": 0.9,
    }
    wev2 = {
        "trigram_family_name": "WEV",
        "trigram": "WME",
        "reference_trigram": "WEV",
        "scaled_target_score": 0.7,
        "scaled_estimated_score": 0.5,
        "estimated_score": 1.1,
    }

    sample_data: list[TrigramScoreDict] = [qxd1, qxd2, sdf1, sdf2, wev1, wev2]

    def test_simple(self):

        groups = group_trigram_scores(self.sample_data)
        assert len(groups) == 3
        # The order of keys is insertion order
        assert list(groups.keys()) == ["QXD", "SDF", "WEV"]
        # The order of trigrams score dictionaries is the same as in the input iterable
        assert groups["SDF"] == [self.sdf1, self.sdf2]
        assert groups["QXD"] == [self.qxd1, self.qxd2]
        assert groups["WEV"] == [self.wev1, self.wev2]

    def test_with_sorting(self):

        groups = group_trigram_scores(self.sample_data, group_sort_by=average_abs_error)

        errors = [average_abs_error(g) for g in groups.values()]

        assert errors[0] < errors[1] < errors[2]


class TestOnehandBaseScore:

    def test_bigram_scores(self, hands_full: Hands):
        indices, _ = hands_full.where("sdf")
        bigram_scores = {
            (11, 8): 1.6,  # SD
            (8, 5): 2.0,  # DF
        }
        params = TrigramModelParameters()
        d = get_onehand_base_score(indices, hands_full.left, bigram_scores, params)
        assert d["ngram1_score"] == 1.6  # SD
        assert d["ngram2_score"] == 2.0  # DF
        assert d["base_score"] == (3 / 4) * (1.6 + 2.0)
        assert d["onehand_base_score"] == d["base_score"]
        assert d["onehand_extra"] == 1.0

    def test_vert2u_sfb_in_onehand(self, hands_full: Hands):
        indices, _ = hands_full.where("evt")
        bigram_scores = {
            (7, 6): 1.6,  # EV
            (6, 1): 2.0,  # VT
        }
        sfb_in_onehand_coeff = 1.05
        vert2u_coeff = 1.12
        params = TrigramModelParameters(
            sfb_in_onehand_coeff=sfb_in_onehand_coeff, vert2u_coeff=vert2u_coeff
        )
        d = get_onehand_base_score(indices, hands_full.left, bigram_scores, params)

        assert d["sft"] == 1.0  # not SFT
        assert d["sfb_in_onehand"] == sfb_in_onehand_coeff
        assert d["vert2u"] == vert2u_coeff
        assert d["onehand_extra"] == vert2u_coeff * sfb_in_onehand_coeff
        assert (
            d["onehand_base_score"]
            == d["base_score"] * vert2u_coeff * sfb_in_onehand_coeff
        )

    def test_sft_vert2u(self, hands_full: Hands):
        indices, _ = hands_full.where("RBT")
        bigram_scores = {
            (4, 3): 1.6,  # RB
            (3, 1): 2.0,  # BT
        }
        sfb_in_onehand_coeff = 1.05
        vert2u_coeff = 1.12
        sft_coeff = 3.4
        params = TrigramModelParameters(
            sft_coeff=sft_coeff,
            sfb_in_onehand_coeff=sfb_in_onehand_coeff,
            vert2u_coeff=vert2u_coeff,
        )
        d = get_onehand_base_score(indices, hands_full.left, bigram_scores, params)

        assert d["sft"] == sft_coeff
        assert d["onehand_extra"] == sft_coeff * vert2u_coeff
        assert d["onehand_base_score"] == d["base_score"] * sft_coeff * vert2u_coeff
