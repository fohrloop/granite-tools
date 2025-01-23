from granite_tools.unigram_scores import create_data_for_fitting


class TestCreateDataForFitting:

    def test_simple(self):

        used_key_indices = [1, 2, 3, 4]

        scores = {
            (1, 2): 0.1,
            (1,): 0.2,
            (3,): 0.3,
            (4,): 0.4,
        }
        X, y = create_data_for_fitting(used_key_indices, scores)
        assert X.shape == (4, 4)
        assert y.shape == (4,)

        assert list(y) == [0.1, 0.2, 0.3, 0.4]

        assert list(X[0]) == [1, 1, 0, 0]
        assert list(X[1]) == [1, 0, 0, 0]
        assert list(X[2]) == [0, 0, 1, 0]
        assert list(X[3]) == [0, 0, 0, 1]
