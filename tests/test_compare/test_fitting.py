from granite_tools.compare.fitting import get_scores


class TestGetScores:

    def test_simple(self):

        comparison_data = [
            ((3,), (2,)),
            ((2,), (1,)),
            ((1,), (66,)),
        ]

        scores = get_scores(comparison_data)

        assert len(scores) == 4
        assert scores[(3,)] > scores[(2,)]
        assert scores[(2,)] > scores[(1,)]
        assert scores[(1,)] > scores[(66,)]
