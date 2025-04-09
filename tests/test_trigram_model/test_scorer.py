from __future__ import annotations

import typing

import pytest

from granite_tools.trigram_features import TrigramFeatures
from granite_tools.trigram_model.scorer import (
    TrigramModelParameters,
    _to_alternating_key_sequences,
    _to_balanced_key_sequences,
    calculate_alternating_score,
    calculate_balanced_score,
    calculate_onehand_score,
    get_alternating_score,
    get_balanced_score,
    get_feature_multiplier,
    get_normalization_factor,
    get_onehand_base_score,
    get_onehand_score,
    get_trigram_score,
)

if typing.TYPE_CHECKING:
    from granite_tools.config import Config
    from granite_tools.hands import Hands


@pytest.fixture
def model_params() -> TrigramModelParameters:
    return TrigramModelParameters()


def get_test_params() -> TrigramModelParameters:
    c_v2u = 0.45
    c_sfb = 1.23
    c_sfs = 1.077
    c_sfsb = 1.044
    c_sft = 3.521
    c_sftb = 2.45
    c_redir = 1.543
    c_eas = 0.34
    c_bal = 0.44
    c_alt = 0.55

    params = TrigramModelParameters(
        easy_rolling_coeff=c_eas,
        balanced_coeff=c_bal,
        alternating_coeff=c_alt,
        redir_coeff=c_redir,
        vert2u_coeff=c_v2u,
        sfb_in_onehand_coeff=c_sfb,
        sft_coeff=c_sft,
        sftb_coeff=c_sftb,
        sfs_coeff=c_sfs,
        sfsb_coeff=c_sfsb,
    )
    return params


class TestGetTrigramScore:

    def test_easy_rolling_onehand(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sdf"
        bigram1 = 1.6
        bigram2 = 1.4
        skipgram = 4.44
        bigram_scores = {
            (11, 8): bigram1,  # SD
            (8, 5): bigram2,  # DF
            (11, 5): skipgram,  # SF
        }

        params = model_params.model_copy()
        params.easy_rolling_coeff = 0.234

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )
        exp_feature_multiplier = params.easy_rolling_coeff
        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["feature_multiplier"] == exp_feature_multiplier
        assert scoredct["score"] == exp_feature_multiplier * expected_base_score

    def test_outward_rolling_onehand_with_sfb(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "hmp"
        bigram1 = 1.6
        bigram2 = 1.4
        skipgram = 3.433

        bigram_scores = {
            (2, 6): bigram1,  # HM
            (6, 13): bigram2,  # MP
            (2, 13): skipgram,  # HP
        }

        params = model_params.model_copy()
        params.sfb_in_onehand_coeff = 1.06

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )

        # This one has also v1x (the bigram MP has 2u jump)
        vert2u_multiplier = 1 + params.vert2u_coeff
        expected_multiplier = params.sfb_in_onehand_coeff * vert2u_multiplier
        score = expected_multiplier * expected_base_score
        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["feature_multiplier"] == expected_multiplier
        assert scoredct["sfp_multiplier"] == params.sfb_in_onehand_coeff
        assert scoredct["vert2u_multiplier"] == vert2u_multiplier
        assert scoredct["score"] == score

    def test_onehand_samecol_with_sft(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "tfv"
        bigram1 = 1.34
        bigram2 = 1.56
        skipgram = 2.22
        bigram_scores = {
            (1, 5): bigram1,  # TF
            (5, 6): bigram2,  # FV
            (1, 6): skipgram,  # TV
        }

        params = model_params.model_copy()
        params.sft_coeff = 1.4

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )
        expected_multiplier = params.sft_coeff
        score = expected_multiplier * expected_base_score

        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["feature_multiplier"] == expected_multiplier
        assert scoredct["sfp_multiplier"] == expected_multiplier
        assert scoredct["score"] == score
        assert scoredct["trigramtype"] == "onehand"  # samecols are also onehand

    def test_onehand_with_180_deg(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "dfs"
        bigram1 = 1.4
        bigram2 = 2.0
        skipgram = 1.6
        bigram_scores = {
            (8, 5): bigram1,  # DF
            (5, 11): bigram2,  # FS
            (8, 11): skipgram,  # DS
        }
        params = model_params.model_copy()
        params.redir_coeff = 3.0

        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )
        expected_score = params.redir_coeff * expected_base_score

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["skipgram_score"] == skipgram
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["score"] == expected_score

    def test_onehand_with_redir_on_ring_finger(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "dwr"
        bigram1 = 1.4
        bigram2 = 2.0
        skipgram = 1.6
        bigram_scores = {
            (8, 10): 1.4,  # DW
            (10, 4): 2.0,  # WR
            (8, 4): 1.6,  # DR
        }
        params = model_params.model_copy()
        params.redir_coeff = 100.0

        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        expected_score = params.redir_coeff * expected_base_score
        assert scoredct["ngram1_score"] == 1.4
        assert scoredct["ngram2_score"] == 2.0
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["redir_multiplier"] == 100.0
        assert scoredct["feature_multiplier"] == 100.0
        assert scoredct["score"] == expected_score
        assert scoredct["trigramtype"] == "onehand"  # redirs are also onehand.

    def test_onehand_with_vert2u(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "wcr"
        bigram1 = 1.44
        bigram2 = 2.0
        skipgram = 1.877

        bigram_scores = {
            (10, 9): bigram1,  # WC
            (9, 4): bigram2,  # CR
            (10, 4): skipgram,  # WR
        }
        params = model_params.model_copy()
        params.redir_coeff = 2.0
        params.vert2u_coeff = 0.787

        expected_base_score = calculate_onehand_score(
            bigram1, bigram2, skipgram, params.w_ac_one
        )

        vert2u = 1 + 2 * params.vert2u_coeff
        expected_score = vert2u * expected_base_score

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )

        assert scoredct["ngram1_score"] == bigram1
        assert scoredct["ngram2_score"] == bigram2
        assert scoredct["skipgram_score"] == skipgram
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["vert2u_multiplier"] == vert2u
        assert scoredct["feature_multiplier"] == vert2u
        assert scoredct["score"] == pytest.approx(expected_score)

    def test_balanced_llr(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sdu"
        bigram_score = 1.6
        unigram_score = 1.1
        bigram_scores = {
            (11, 8): bigram_score,  # SD
            (4,): unigram_score,  # U
        }

        params = model_params.model_copy()
        params.balanced_coeff = 3.5

        expected_base_score, expected_score = calculate_balanced_score(
            bigram_score, unigram_score, params
        )

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["trigramtype"] == "balanced"

    def test_balanced_lrr(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "sku"
        bigram_score = 1.6
        unigram_score = 1.1
        bigram_scores = {
            (11,): unigram_score,  # S
            (8, 4): bigram_score,  # KU
        }

        params = model_params.model_copy()
        params.balanced_coeff = 3.5

        expected_base_score, expected_score = calculate_balanced_score(
            bigram_score, unigram_score, params
        )
        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score

    def test_balanced_rrl(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "lkr"
        bigram_score = 1.4
        unigram_score = 1.2
        bigram_scores = {
            (11, 8): bigram_score,  # LK
            (4,): unigram_score,  # R
        }

        params = model_params.model_copy()
        params.balanced_coeff = 3.5

        expected_base_score, expected_score = calculate_balanced_score(
            bigram_score, unigram_score, params
        )

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score

    def test_balanced_rll(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):

        ngram = "ldr"
        bigram_score = 1.4
        unigram_score = 1.5
        bigram_scores = {
            (11,): unigram_score,  # L
            (8, 4): bigram_score,  # DR
        }

        params = model_params.model_copy()
        params.balanced_coeff = 3.5

        expected_base_score, expected_score = calculate_balanced_score(
            bigram_score, unigram_score, params
        )
        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score

    def test_balanced_with_space(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "_ku"
        bigram_score = 1.4
        unigram_score = 1.02
        bigram_scores = {
            (17,): unigram_score,  # " " (space. inner thumb key on left)
            (8, 4): bigram_score,  # ku
        }

        params = model_params.model_copy()
        params.balanced_coeff = 2.0
        expected_base_score, expected_score = calculate_balanced_score(
            bigram_score, unigram_score, params
        )
        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score

    def test_balanced_left_left_untypable(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "sd("  # ( is not in the layout
        bigram_scores = {
            (11, 8): 1.6,  # SD
        }

        params = model_params.model_copy()
        params.balanced_coeff = 1.4

        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == 0

    def test_alternating(self, model_params: TrigramModelParameters, hands_full: Hands):
        ngram = "skf"
        bigram_score = 1.4
        unigram_score = 1.01
        bigram_scores = {
            (11, 5): bigram_score,  # SF
            (8,): unigram_score,  # K
        }
        params = model_params.model_copy()
        params.u_weight_alternating = 0.12
        params.alternating_coeff = 1.045

        expected_base_score, expected_score = calculate_alternating_score(
            bigram_score, unigram_score, params
        )
        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["u_weight_alternating"] == params.u_weight_alternating
        assert scoredct["alternating_coeff"] == params.alternating_coeff

    def test_alternating_with_thumb(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        ngram = "l_j"
        bigram_score = 4.4
        unigram_score = 1.2
        bigram_scores = {
            (11, 5): bigram_score,  # LJ
            (17,): unigram_score,  # " " (space; inner thumb key on left)
        }
        params = model_params.model_copy()
        params.u_weight_alternating = 0.23
        params.alternating_coeff = 1.045

        expected_base_score, expected_score = calculate_alternating_score(
            bigram_score, unigram_score, params
        )
        scoredct = get_trigram_score(
            ngram=ngram,
            hands=hands_full,
            params=params,
            bigram_scores=bigram_scores,
        )
        assert scoredct["score"] == expected_score
        assert scoredct["base_score"] == expected_base_score
        assert scoredct["u_weight_alternating"] == params.u_weight_alternating
        assert scoredct["alternating_coeff"] == params.alternating_coeff


class TestToKeySequences:

    def test_llr(self):
        assert _to_balanced_key_sequences([1, 2, 3], ["Left", "Left", "Right"]) == (
            ((1, 2), (3,)),
            ("Left", "Right"),
        )

    def test_rll(self):
        assert _to_balanced_key_sequences([4, 2, 3], ["Right", "Left", "Left"]) == (
            ((4,), (2, 3)),
            ("Right", "Left"),
        )


class TestToAlternatingKeySequences:
    def test_lrl(self):
        assert _to_alternating_key_sequences([1, 2, 3], ["Left", "Right", "Left"]) == (
            ((1, 3), (2,)),
            ("Left", "Right"),
        )

    def test_rlr(self):
        assert _to_alternating_key_sequences([1, 2, 3], ["Right", "Left", "Right"]) == (
            ((1, 3), (2,)),
            ("Right", "Left"),
        )


class TestGetAlternatingScore:

    indices = (1, 2, 3)

    ab = 2.123
    bc = 1.512
    ac = 1.89
    a = 1.1
    b = 1.2
    c = 1.3

    bigram_scores = {
        (1,): a,
        (2,): b,
        (3,): c,
        (1, 2): ab,
        (2, 3): bc,
        (1, 3): ac,
    }

    params = get_test_params()

    def test_lrl(self):
        keytypes = ("Left", "Right", "Left")
        params = self.params.model_copy()
        params.u_weight_alternating = 0.777

        scoredct = get_alternating_score(
            self.indices, keytypes, self.bigram_scores, params
        )

        unigram_keyseq = (2,)  # Right
        bigram_keyseq = (1, 3)  # Left - Left

        bigram_score = self.bigram_scores[bigram_keyseq]
        unigram_score = self.bigram_scores[unigram_keyseq]

        base_score = params.u_weight_alternating * unigram_score + bigram_score

        assert scoredct["ngram1_score"] == bigram_score  # ngram1 is always the bigram
        assert scoredct["ngram2_score"] == unigram_score  # ngram2 is always the unigram
        assert scoredct["trigramtype"] == "alternating"
        assert scoredct["base_score"] == base_score
        assert scoredct["score"] == params.alternating_coeff * base_score


class TestGetBalancedScore:

    indices = (1, 2, 3)

    ab = 2.123
    bc = 1.512
    ac = 1.89
    a = 1.1
    b = 1.2
    c = 1.3

    bigram_scores = {
        (1,): a,
        (2,): b,
        (3,): c,
        (1, 2): ab,
        (2, 3): bc,
        (1, 3): ac,
    }

    params = get_test_params()

    def test_lrr(self):
        keytypes = ("Left", "Right", "Right")

        scoredct = get_balanced_score(
            self.indices, keytypes, self.bigram_scores, self.params
        )

        unigram_keyseq = (1,)  # Left
        bigram_keyseq = (2, 3)  # Right, Right

        bigram_score = self.bigram_scores[bigram_keyseq]
        unigram_score = self.bigram_scores[unigram_keyseq]

        base_score = self.params.u_weight_balanced * unigram_score + bigram_score

        assert scoredct["ngram1_score"] == bigram_score  # ngram1 is always the bigram
        assert scoredct["ngram2_score"] == unigram_score  # ngram2 is always the unigram
        assert scoredct["trigramtype"] == "balanced"
        assert scoredct["base_score"] == base_score
        assert scoredct["score"] == self.params.balanced_coeff * base_score

    def test_llr(self):
        keytypes = ("Left", "Left", "Right")

        scoredct = get_balanced_score(
            self.indices, keytypes, self.bigram_scores, self.params
        )

        bigram_keyseq = (1, 2)  # Left, Left
        unigram_keyseq = (3,)  # Right

        bigram_score = self.bigram_scores[bigram_keyseq]
        unigram_score = self.bigram_scores[unigram_keyseq]
        base_score = self.params.u_weight_balanced * unigram_score + bigram_score
        assert scoredct["ngram1_score"] == bigram_score  # ngram1 is always the bigram
        assert scoredct["ngram2_score"] == unigram_score  # ngram2 is always the unigram
        assert scoredct["trigramtype"] == "balanced"
        assert scoredct["base_score"] == base_score
        assert scoredct["score"] == self.params.balanced_coeff * base_score


class TestGetOnehandScore:

    ab = 2.123
    bc = 1.512
    ac = 1.89

    bigram_scores = {
        (1, 2): ab,
        (2, 3): bc,
        (1, 3): ac,
    }
    params = get_test_params()

    indices = (1, 2, 3)

    def test_simple(self):

        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFB",
            vert2u="v2x",
            redir=None,
            easy_rolling=None,
        )

        scoredct = get_onehand_score(
            self.indices, self.bigram_scores, features, self.params
        )

        m = get_feature_multiplier(features, self.params)
        d = get_onehand_base_score(self.indices, self.bigram_scores, self.params)

        assert scoredct == {
            "feature_multiplier": m["feature_multiplier"],
            "easy_rolling_multiplier": m["easy_rolling_multiplier"],
            "vert2u_multiplier": m["vert2u_multiplier"],
            "sfp_multiplier": m["sfp_multiplier"],
            "redir_multiplier": m["redir_multiplier"],
            "ngram1_score": d["ngram1_score"],
            "ngram2_score": d["ngram2_score"],
            "skipgram_score": d["skipgram_score"],
            "base_score": d["base_score"],
            "score": d["base_score"] * m["feature_multiplier"],
            "trigramtype": "onehand",
        }


class TestOnehandFeatureMultiplier:
    c_v2u = 0.45
    c_sfb = 1.23
    c_sfs = 1.077
    c_sfsb = 1.044
    c_sft = 3.521
    c_sftb = 2.45
    c_redir = 1.543
    c_eas = 0.34
    c_bal = 0.44
    c_alt = 0.55

    params = TrigramModelParameters(
        easy_rolling_coeff=c_eas,
        balanced_coeff=c_bal,
        alternating_coeff=c_alt,
        redir_coeff=c_redir,
        vert2u_coeff=c_v2u,
        sfb_in_onehand_coeff=c_sfb,
        sft_coeff=c_sft,
        sftb_coeff=c_sftb,
        sfs_coeff=c_sfs,
        sfsb_coeff=c_sfsb,
    )

    def test_no_features(self):

        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern=None,
            vert2u=None,
            redir=None,
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)
        assert d["feature_multiplier"] == 1.0

    def test_easy_rolling(self):
        features = TrigramFeatures(
            main_type="onehand",
            easy_rolling="easy-rolling",
        )

        d = get_feature_multiplier(features, self.params)
        assert d["feature_multiplier"] == self.c_eas
        assert d["easy_rolling_multiplier"] == self.c_eas
        assert d["vert2u_multiplier"] == 1.0
        assert d["sfp_multiplier"] == 1.0
        assert d["redir_multiplier"] == 1.0

    def test_sfb_v2x(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFB",
            vert2u="v2x",
            redir=None,
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)

        vert2u = 1.0 + 2 * self.c_v2u
        assert d["feature_multiplier"] == vert2u * self.c_sfb
        assert d["vert2u_multiplier"] == vert2u
        assert d["sfp_multiplier"] == self.c_sfb

    def test_redir_sfs(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFS",
            vert2u=None,
            redir="redir",
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)

        assert d["feature_multiplier"] == self.c_sfs * self.c_redir
        assert d["sfp_multiplier"] == self.c_sfs
        assert d["redir_multiplier"] == self.c_redir

    def test_sft(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFT",
            vert2u=None,
            redir=None,
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)

        assert d["feature_multiplier"] == self.c_sft
        assert d["sfp_multiplier"] == self.c_sft
        assert d["redir_multiplier"] == 1.0

    def test_redir_sfs_v1x(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFS",
            vert2u="v1x",
            redir="redir",
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)
        vert2u = 1.0 + self.c_v2u
        assert d["feature_multiplier"] == vert2u * self.c_sfs * self.c_redir
        assert d["vert2u_multiplier"] == vert2u
        assert d["sfp_multiplier"] == self.c_sfs
        assert d["redir_multiplier"] == self.c_redir

    def test_sfsb(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFSb",
            vert2u=None,
            redir=None,
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)
        assert d["feature_multiplier"] == self.c_sfsb
        assert d["vert2u_multiplier"] == 1.0
        assert d["sfp_multiplier"] == self.c_sfsb
        assert d["redir_multiplier"] == 1.0

    def test_sftb(self):
        features = TrigramFeatures(
            main_type="onehand",
            single_finger_pattern="SFTb",
            vert2u=None,
            redir=None,
            easy_rolling=None,
        )

        d = get_feature_multiplier(features, self.params)
        assert d["feature_multiplier"] == self.c_sftb
        assert d["vert2u_multiplier"] == 1.0
        assert d["sfp_multiplier"] == self.c_sftb
        assert d["redir_multiplier"] == 1.0


class TestOnehandBaseScore:

    def test_simple(self):
        ab = 1.0
        bc = 2.0
        ac = 3.0

        bigram_scores = {
            (1, 2): ab,
            (2, 3): bc,
            (1, 3): ac,
        }
        params = TrigramModelParameters()
        params.w_ac_one = 1.567
        indices = (1, 2, 3)
        a_norm = get_normalization_factor(params.w_ac_one)

        d = get_onehand_base_score(indices, bigram_scores, params)
        assert d["ngram1_score"] == ab
        assert d["ngram2_score"] == bc
        assert d["skipgram_score"] == ac
        assert d["base_score"] == a_norm * (ab + bc + params.w_ac_one * ac)


class TestGetNormalizationFactor:

    def test_simple(self):

        assert get_normalization_factor(1.0) == 3 / 6
        assert get_normalization_factor(0.65) == 30 / 53
        assert get_normalization_factor(0) == 3 / 4
