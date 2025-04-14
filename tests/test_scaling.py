from __future__ import annotations

from granite_tools.scaling import get_scaled_scores


class TestGetScaledBigramScores:

    def test_simple(self):
        bigram_scores = {
            (1, 2): 1.0,
            (2, 3): 3.0,  # halfway from 1 to 5
            (3, 4): 2.0,  # 1/4 from 1 to 4
            (4, 5): 4.0,  # 3/4 from 1 to 4
            (5, 6): 5.0,  # max
        }
        scaled_scores = get_scaled_scores(bigram_scores, newmax=10.0)
        expected_scores = {
            (1, 2): 1.0,
            (2, 3): 5.5,  # halfway from 1 to 10
            (3, 4): 1 + 4.5 / 2,  # 1/4 from 1 to 10
            (4, 5): 1 + 4.5 * 1.5,  # 3/4 from 1 to 10
            (5, 6): 10.0,  # max
        }
        assert scaled_scores == expected_scores

        scaled_scores = get_scaled_scores(bigram_scores, newmax=10.0)
        expected_scores = {k: v for k, v in expected_scores.items()}
        assert scaled_scores == expected_scores
