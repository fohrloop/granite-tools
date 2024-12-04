from textwrap import dedent
from typing import Sequence
from unittest.mock import mock_open, patch

from granite_tools.config import Config
from granite_tools.hands import Hands, get_hands_data
from granite_tools.scorer.data import get_trigram_data
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    get_score,
    load_trigram_scores,
)


class TestGetTrigramData:

    def test_simple(self, config_full: Config):

        trigram_score_filecontents = """
        [VCX]
        MCX = 3.0
        CXV = 4.0

        [SEF]
        SFE = 2.0

        [XCV]
        "XCM" = 1.5
        "XV," = 2.0
        """
        trigram_scores = self.get_test_trigram_scores(trigram_score_filecontents)
        hands = get_hands_data(config_full)
        bigram_scores = self.create_fake_bigram_scores(trigram_scores, hands)
        df = get_trigram_data(config_full, bigram_scores, trigram_scores)

        assert len(df) == 5  # 3 sets of trigrams.(3+2+3)

        params = TrigramModelParameters.from_config(config_full)
        for trigram in ("MCX", "CXV"):
            ref_trigram = "VCX"
            assert df.loc[trigram].reference_trigram == ref_trigram
            assert df.loc[trigram].trigram_family_name == "XCV"

            ref_score = get_score(ref_trigram, hands, params, bigram_scores)["score"]
            score = get_score(trigram, hands, params, bigram_scores)
            scaled_score = score["score"] / ref_score
            assert df.loc[trigram].scaled_estimated_score == scaled_score

            target_score = trigram_scores[ref_trigram][trigram]
            assert df.loc[trigram].scaled_target_score == target_score

        for trigram in ("SFE",):
            ref_trigram = "SEF"
            assert df.loc[trigram].reference_trigram == ref_trigram
            assert df.loc[trigram].trigram_family_name == "SEF"

            ref_score = get_score(ref_trigram, hands, params, bigram_scores)["score"]
            score = get_score(trigram, hands, params, bigram_scores)
            scaled_score = score["score"] / ref_score
            assert df.loc[trigram].scaled_estimated_score == scaled_score

            target_score = trigram_scores[ref_trigram][trigram]
            assert df.loc[trigram].scaled_target_score == target_score
        for trigram in ("XCM", "XV,"):
            ref_trigram = "XCV"
            assert df.loc[trigram].reference_trigram == ref_trigram
            assert df.loc[trigram].trigram_family_name == "XCV"

            ref_score = get_score(ref_trigram, hands, params, bigram_scores)["score"]
            score = get_score(trigram, hands, params, bigram_scores)
            scaled_score = score["score"] / ref_score
            assert df.loc[trigram].scaled_estimated_score == scaled_score

            target_score = trigram_scores[ref_trigram][trigram]
            assert df.loc[trigram].scaled_target_score == target_score

    def create_fake_bigram_scores(
        self, trigram_scoredct: dict[str, dict[str, float]], hands: Hands
    ):
        def iter_trigrams():
            for ref_trigram, trigram_set in trigram_scoredct.items():
                yield ref_trigram
                for trigram in trigram_set:
                    yield trigram

        bigram_scores = {}
        for trigram in iter_trigrams():
            indices, _ = hands.where(trigram)

            bigram_scores[indices[0], indices[1]] = 2.0
            bigram_scores[indices[1], indices[2]] = 2.0
            for idx in indices:
                bigram_scores[(idx,)] = 1.0

        return bigram_scores

    def get_test_trigram_scores(self, trigram_score_filecontents: str):

        trigram_scoredata = dedent(trigram_score_filecontents.strip("\n"))

        with patch(
            "builtins.open", mock_open(read_data=trigram_scoredata.encode("utf-8"))
        ):
            trigram_scores = load_trigram_scores("____fooooo____")

        return trigram_scores
