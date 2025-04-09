from __future__ import annotations

from functools import partial

from granite_tools.hands import Hands
from granite_tools.score_ratios import create_score_ratio_entries


class TestCreateScoreRatioEntries:

    def test_simple(self, hands_full: Hands):

        input_data = [
            {"ref": "ABC", "ngram": "DEF", "score_ratio": 1.5},
            {"ref": "EFG", "ngram": "HIJ", "score_ratio": 2.2},
        ]

        where = partial(hands_full.where, visualization=True)
        get_keyseq = lambda x: where(x)[0]

        expected_output = [
            {
                "ref": "ABC",
                "ngram": "DEF",
                "score_ratio": 1.5,
                "ref_keyseq": get_keyseq("ABC"),
                "ngram_keyseq": get_keyseq("DEF"),
            },
            {
                "ref": "EFG",
                "ngram": "HIJ",
                "score_ratio": 2.2,
                "ref_keyseq": get_keyseq("EFG"),
                "ngram_keyseq": get_keyseq("HIJ"),
            },
        ]
        assert create_score_ratio_entries(input_data, hands_full) == expected_output
