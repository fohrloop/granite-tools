import re

import pytest

from granite_tools.app_types import KeySeq
from granite_tools.bigram_compare.comparisons import keyseqs_to_comparisons
from granite_tools.bigram_compare.fitting import get_scores
from granite_tools.bigram_compare.merge_data import merge_compare_to_rankings
from granite_tools.bigram_compare.scorer import CompareDataDict, DataValidityError


class TestMergeCompareToRankings:
    def test_added_key_sequences(self):
        old_compare, new_rankings, expected = self.create_test_data()

        out = merge_compare_to_rankings(old_compare, new_rankings)
        assert out == expected

    def test_removed_key_sequences(self):
        old_compare, new_rankings, _ = self.create_test_data()
        new_rankings = [(3,), (4,)]  # removing 1 and 2
        with pytest.raises(
            DataValidityError,
            match=re.escape(
                "comparisons_all contains pair ((4,), (1,)) which has key sequence (1,) that is not part of initial_order"
            ),
        ):
            merge_compare_to_rankings(old_compare, new_rankings)

    @staticmethod
    def create_test_data() -> tuple[CompareDataDict, list[KeySeq]]:
        # Creates old ".compare.pickle" data and "new rankings" data
        initial_order = [
            (2,),
            (1,),
            (3,),
            (4,),
        ]
        # 5 and 6 added.
        new_initial_order = [
            (2,),
            (5,),
            (1,),
            (3,),
            (4,),
            (6,),
        ]
        current_order = [
            (2,),
            (3,),
            (1,),
            (4,),
        ]
        processed_key_sequences = [
            (4,),
            (1,),
            (2,),
        ]  # (3,) is not yet processed

        initial_comparisons = keyseqs_to_comparisons(initial_order)
        manual_comparisons = [
            ((4,), (1,)),  # 4
            ((4,), (3,)),  # 4
            ((4,), (2,)),  # 4
            ((1,), (2,)),  # 1
            ((1,), (3,)),  # 1
            ((4,), (1,)),  # 1
            ((2,), (3,)),  # 2
            ((1,), (2,)),  # 2
            ((4,), (2,)),  # 2
        ]
        comparisons_all = initial_comparisons + manual_comparisons
        current_scores = get_scores(comparisons_all)
        # check that the generated data is okay
        expected_current_order = sorted(initial_order, key=lambda x: current_scores[x])
        assert current_order == expected_current_order

        compare = dict(
            comparisons_all=comparisons_all,
            initial_order=initial_order,
            current_order=current_order,
            processed_key_sequences=processed_key_sequences,
            current_scores=current_scores,
            p95_fraction=1 / 4,
            pairs_per_sequence=3,
        )

        # You extect the have the same manual comparisons but with different automatic
        # comparisons in the beginning (from the initial ordering)
        expected_new_comparisons_all = (
            keyseqs_to_comparisons(new_initial_order) + manual_comparisons
        )

        expected_new_current_scores = get_scores(expected_new_comparisons_all)
        expected_new_current_order = sorted(
            new_initial_order, key=lambda x: expected_new_current_scores[x]
        )
        expected = dict(
            comparisons_all=expected_new_comparisons_all,
            initial_order=new_initial_order,
            current_order=expected_new_current_order,
            processed_key_sequences=processed_key_sequences,  # same
            current_scores=expected_new_current_scores,
            p95_fraction=compare["p95_fraction"],
            pairs_per_sequence=compare["pairs_per_sequence"],
        )

        return (compare, new_initial_order, expected)
