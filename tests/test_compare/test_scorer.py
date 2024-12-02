import itertools
import random
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from granite_tools.compare.comparisons import keyseqs_to_comparisons
from granite_tools.compare.scorer import (
    ComparisonBasedScorer,
    DataValidityError,
    RoundFinishedError,
    filter_candidates,
    get_distribution_sigma,
    is_bigram,
    is_repeat,
    is_unigram,
    unique_numbers_from_normal_dist,
    verify_data,
)

this_folder = Path(__file__).parent


def create_permutations(idx_range: int, sequence_lengths: tuple[int, ...] = (1, 2)):
    key_indices = list(range(idx_range))
    permutations_lst = []
    for seq_length in sequence_lengths:
        for seq in itertools.product(key_indices, repeat=seq_length):
            permutations_lst.append(seq)
    return permutations_lst


class TestComparisonBasedScorer:

    ordered_data = create_permutations(16)

    def test_initialization(self):
        random.seed(42)

        scorer = ComparisonBasedScorer(self.ordered_data)
        assert len(scorer.current_comparison_pairs) == 10

        # The pairs are formed well
        for pair in scorer.current_comparison_pairs:
            for ks in pair:
                assert ks in self.ordered_data
            assert pair[0] == scorer.current_key_sequence
            assert pair[1] != scorer.current_key_sequence

        # The comparisons is empty
        assert scorer.comparisons_new == []

    def test_selecting_left_and_right(self):
        random.seed(42)

        scorer = ComparisonBasedScorer(self.ordered_data)
        expected_first_pair = tuple(reversed(scorer.current_comparison_pair))

        # Make one comparison
        scorer.handle_select_left()

        # The comparisons is not empty
        assert len(scorer.comparisons_new) == 1
        expected_second_pair = scorer.current_comparison_pair
        # Selected left side having SMALLER score (put selected on the right
        # as in choix the order is (larger,smaller))
        assert scorer.comparisons_new[0][0] != scorer.current_key_sequence
        assert scorer.comparisons_new[0][1] == scorer.current_key_sequence
        assert scorer.comparisons_new[0] == expected_first_pair

        # Make another comparison
        scorer.handle_select_right()
        assert len(scorer.comparisons_new) == 2
        assert scorer.comparisons_new[0] == expected_first_pair
        assert scorer.comparisons_new[1] == expected_second_pair

        # Selected right side having larger SMALLER score (cost)
        assert scorer.comparisons_new[1][0] == scorer.current_key_sequence
        assert scorer.comparisons_new[1][1] != scorer.current_key_sequence

        # go back
        scorer.handle_goto_previous()
        assert scorer.comparisons_new == [expected_first_pair]
        assert scorer.current_comparison_pair == expected_second_pair

    def test_cannot_place_more_than_max(self):
        random.seed(42)

        scorer = ComparisonBasedScorer(self.ordered_data)
        assert len(scorer.current_comparison_pairs) == 10

        for i in range(10):
            scorer.handle_select_left() if i % 2 else scorer.handle_select_right()

        with pytest.raises(RoundFinishedError):
            scorer.handle_select_left()
        with pytest.raises(RoundFinishedError):
            scorer.handle_select_right()

    def test_fitting(self):
        random.seed(42)
        np.random.seed(42)

        scorer = ComparisonBasedScorer(self.ordered_data)
        scorer.fit(allow_unfinished=True)
        initial_order = scorer.current_order
        initial_scores = scorer.current_scores

        first_pair = scorer.current_comparison_pair
        keyseq = first_pair[0]

        for i in range(10):
            # Always select left (decreasing it's score)
            scorer.handle_select_left()

        assert scorer.is_current_round_finished() == True
        comparisons = scorer.comparisons_new

        scorer.fit()
        new_order = scorer.current_order
        new_scores = scorer.current_scores
        assert initial_scores != new_scores
        assert initial_order != new_order
        # The winner (left) score (cost) DECREASED
        assert new_scores[keyseq] < initial_scores[keyseq]

        # Scores for all other key sequences in the comparisons have been
        # changed. If the keysequence is close to the "current keysequence",
        # the score might have increased. Otherwise, it should have decreased.
        # Since we don't know how the score should have changed, we can only
        # check that the score has changed.
        for _, keyseq_other in comparisons:
            assert new_scores[keyseq_other] != initial_scores[keyseq_other]

    @pytest.fixture
    def tempfile(self):
        file = this_folder / "_temptile_test_saving_and_loading_"
        yield str(file)
        file.unlink()

    def test_saving_and_loading(self, tempfile: str):
        scorer = ComparisonBasedScorer(self.ordered_data)

        for _ in range(10):
            scorer.handle_select_left()

        dct1 = scorer.to_dict()
        scorer.fit()
        scorer.select_next_key_sequence()
        dct2 = scorer.to_dict()

        contant_attrs = (
            "initial_order",
            "p95_fraction",
            "pairs_per_sequence",
        )
        changing_attrs = (
            "processed_key_sequences",
            "current_key_sequence",
            "comparisons_all",
            "current_scores",
            "current_order",
        )
        for key in dct1:
            if key in contant_attrs:
                assert dct1[key] == dct2[key]
            elif key in changing_attrs:
                assert dct1[key] != dct2[key], f"Key: '{key}' should be different"
            else:
                raise ValueError(f"Unexpected key: {key}")

        scorer.save_to_file(tempfile)
        scorer2 = ComparisonBasedScorer.load_from_file(tempfile)

        assert scorer.to_dict() == scorer2.to_dict()

    def test_loading_and_continuing_works(self, tempfile: str):
        # There was a problem when loading from file and continuing the work. Test that
        # case here.

        # Setup
        used_all_key_sequences = None

        def patched_filter_candidates(all_key_sequences, current_key_sequence):
            nonlocal used_all_key_sequences
            used_all_key_sequences = all_key_sequences
            return filter_candidates(all_key_sequences, current_key_sequence)

        # Act
        scorer1 = ComparisonBasedScorer(self.ordered_data)

        for _ in range(10):
            scorer1.handle_select_left()

        scorer1.fit()
        scorer1.save_to_file(tempfile)

        # Now we have our first fit done. Let's load it and continue the work.
        # The pached function will record the key sequences used in the filtering
        with patch("granite_tools.compare.scorer.filter_candidates", patched_filter_candidates):
            scorer2 = ComparisonBasedScorer.load_from_file(tempfile)

        # Check that correct all_key_sequences was used. In particular, when filtering
        # candidates, the initial order should NOT be used, but the current order!
        assert used_all_key_sequences != scorer1.initial_order
        assert used_all_key_sequences == scorer1.current_order

        for _ in range(10):
            scorer2.handle_select_left()

        scorer2.fit()
        scorer2_data_after_fit = scorer2.to_dict()

        # This should not cause any problems! (it did before)
        verify_data(scorer2_data_after_fit)


class TestGetDistributionSigma:

    def test_one_third(self):
        assert get_distribution_sigma(10, 1 / 3) == pytest.approx(0.8503557)


class TestUniqueNumbersFromNormalDist:

    def test_range_too_small(self):
        with pytest.raises(ValueError):
            unique_numbers_from_normal_dist(
                num_samples=3, current_index=1, lower_bound=0, upper_bound=1, sigma=2
            )

    def test_normal(self):
        np.random.seed(42)

        assert unique_numbers_from_normal_dist(
            2, 0, lower_bound=-1, upper_bound=1, sigma=5
        ) == [-1, 1]

    def test_with_realistic_data(self):
        np.random.seed(22)
        n = 270
        sigma = get_distribution_sigma(n, 1 / 4)
        out = unique_numbers_from_normal_dist(
            num_samples=10,
            current_index=100,
            lower_bound=0,
            upper_bound=n,
            sigma=sigma,
        )
        assert sorted(out) == [81, 82, 92, 96, 98, 106, 110, 111, 113, 118]


class TestNgramIsFunctions:

    def test_is_unigram(self):
        assert is_unigram((1,)) == True
        assert is_unigram((1, 2)) == False

    def test_is_repeat(self):
        assert is_repeat((1, 1)) == True
        assert is_repeat((1, 2)) == False

    def test_is_bigram(self):
        assert is_bigram((1, 2)) == True
        assert is_bigram((1, 1)) == False
        assert is_bigram((1,)) == False
        assert is_bigram((1, 2, 3)) == False


class TestFilterCandidates:

    def test_bigram(self):
        # bigrams may be compared with anything
        assert filter_candidates(
            [(1, 2), (1, 1), (2, 2)], current_key_sequence=(1, 2)
        ) == [(1, 2), (1, 1), (2, 2)]

    def test_current_is_not_dropped_out(self):
        assert filter_candidates(
            [(1, 2), (1, 1), (2, 2)], current_key_sequence=(1, 1)
        ) == [(1, 2), (1, 1)]

    def test_unigram_is_filtered(self):
        assert filter_candidates([(2,), (0, 1), (1,)], current_key_sequence=(1,)) == [
            (0, 1),
            (1,),
        ]

    def test_unigrams_and_repeats(self):
        assert filter_candidates(
            [(2,), (0, 1), (0, 2), (1,), (1, 1)], current_key_sequence=(1,)
        ) == [(0, 1), (0, 2), (1,)]


@pytest.fixture
def data():
    return get_data(processed_key_sequences=[(0,), (1,), (2,)])


def get_data(
    processed_key_sequences: list[tuple[int, ...]], pairs_per_sequence=4
) -> dict:

    initial_order = list((i,) for i in range(10))
    comparisons_all = keyseqs_to_comparisons(initial_order)

    for ks in processed_key_sequences:
        candidate_pairs = [seq for seq in initial_order if seq != ks]
        for i in range(pairs_per_sequence):
            ks2 = candidate_pairs[i]
            pair = (ks, ks2) if i % 2 else (ks2, ks)
            comparisons_all.append(pair)

    return dict(
        initial_order=initial_order,
        pairs_per_sequence=pairs_per_sequence,
        processed_key_sequences=processed_key_sequences,
        comparisons_all=comparisons_all,
    )


class TestVerifyData:

    def test_ok(self, data: dict):
        # no problemos
        verify_data(data)

    def test_pairs_per_sequence_too_large(self, data: dict):
        data["pairs_per_sequence"] += 1
        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Number of comparisons (21) does not match the expected number of comparisons (24). Debug info: initial_pairs=9, pairs_per_sequence=5, n_processed_sequences=3"
            ),
        ):
            verify_data(data)

    def test_pairs_per_sequence_too_small(self, data: dict):
        data["pairs_per_sequence"] -= 1
        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Number of comparisons (21) does not match the expected number of comparisons (18). Debug info: initial_pairs=9, pairs_per_sequence=3, n_processed_sequences=3"
            ),
        ):
            verify_data(data)

    def test_processed_key_sequences_extra_item(self, data: dict):
        data["processed_key_sequences"].append((5,))
        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Number of comparisons (21) does not match the expected number of comparisons (25). Debug info: initial_pairs=9, pairs_per_sequence=4, n_processed_sequences=4"
            ),
        ):
            verify_data(data)

    def test_processed_key_sequences_one_missing(self, data: dict):
        data["processed_key_sequences"].pop()
        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Number of comparisons (21) does not match the expected number of comparisons (17). Debug info: initial_pairs=9, pairs_per_sequence=4, n_processed_sequences=2"
            ),
        ):
            verify_data(data)

    def test_wrong_comparison_pair(self, data: dict):

        assert data["comparisons_all"][12] == ((0,), (4,))
        data["comparisons_all"][12] = ((10,), (11,))

        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Key sequence (0,) is not in the comparison pair ((10,), (11,)) at index 12 or comparisons_all!"
            ),
        ):
            verify_data(data)

    def test_duplicates_in_processed_key_seqs(self):
        data = get_data(processed_key_sequences=[(0,), (1,), (0,)])
        with pytest.raises(
            DataValidityError,
            match=re.escape("Duplicate key sequence (0,) in processed_key_sequences!"),
        ):
            verify_data(data)

    def test_missing_key_sequences_in_processed(self):
        data = get_data(processed_key_sequences=[(i,) for i in range(10)])
        data["processed_key_sequences"].pop()
        data["processed_key_sequences"].append((99,))
        new_comparisons_all = []
        for i, pair in enumerate(data["comparisons_all"]):
            if i > len(data["initial_order"]):
                if pair[0] == (9,):
                    pair = ((99,), pair[1])
                if pair[1] == (9,):
                    pair = (pair[0], (99,))
            new_comparisons_all.append(pair)
        data["comparisons_all"] = new_comparisons_all

        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "Key sequences in processed_key_sequences do not match the ones in initial_order!: missing={(9,)} extra={(99,)}"
            ),
        ):
            verify_data(data)

    def test_duplicate_pairs_for_key_sequence(self, data: dict):
        # modifying the last comparison pair so its a duplicate

        data["comparisons_all"][-1] = data["comparisons_all"][-2]
        with pytest.raises(
            DataValidityError,
            match=re.escape("Key sequence (2,) is paired with (3,) more than once!"),
        ):
            verify_data(data)
