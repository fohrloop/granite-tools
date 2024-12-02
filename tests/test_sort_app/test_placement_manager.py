import pytest

from granite_tools.effort import Hand, create_permutations
from granite_tools.sort_app.placement_manager import (
    NgramPlacementManager,
    split_ordered_ngrams_into_two_halfs,
)


class TestPlacementManager:
    left = Hand(hand="Left", symbols={x: str(x) for x in range(7)})
    right = Hand(hand="Right", symbols={x: chr(ord("A") + x) for x in range(7)})
    permutations = create_permutations(left, right, sequence_lengths=(1,))

    def test_basics(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        assert manager.current_ngram == (0,)
        assert manager.ordered_ngrams == []
        assert manager._current_ngram_movement_history == []
        assert manager.left_of_current is None
        assert manager.right_of_current is None
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == []

        manager.place_current_ngram()
        # The previous current ngram was placed
        assert manager.ordered_ngrams == [(0,)]
        # The current ngram is now the next one
        assert manager.current_ngram == (1,)

        # Current is always added to the center. If that's not possible, the
        # right side has more items.
        assert manager.left_of_current is None
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == [(0,)]
        manager.move_right()

        assert manager.left_of_current == (0,)
        assert manager.right_of_current is None
        assert manager.ngrams_left_side_of_current == [(0,)]
        assert manager.ngrams_right_side_of_current == []

        # Move back to left
        manager.move_left()
        assert manager.left_of_current is None
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == [(0,)]

    def test_adding_ngrams(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]

        manager.place_current_ngram()
        # The previous current ngram was placed
        assert manager.ordered_ngrams == [(0,)]
        # The current ngram is now the next one
        assert manager.current_ngram == (1,)

        manager.move_right()
        assert manager.left_of_current == (0,)
        assert manager.right_of_current is None

        manager.place_current_ngram()
        # The previous current ngram was placed (the the right side)
        assert manager.ordered_ngrams == [(0,), (1,)]
        assert manager.current_ngram == (2,)

        # We should be in the middle now
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (1,)
        assert manager.ngrams_left_side_of_current == [(0,)]
        assert manager.ngrams_right_side_of_current == [(1,)]

        # Let's place this one to the middle
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(0,), (2,), (1,)]
        assert manager.current_ngram == (3,)
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == [(0,)]
        assert manager.ngrams_right_side_of_current == [(2,), (1,)]

        # Let's place this one to the left
        manager.move_left()
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(3,), (0,), (2,), (1,)]
        assert manager.current_ngram == (4,)
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == [(3,), (0,)]
        assert manager.ngrams_right_side_of_current == [(2,), (1,)]

        # Let's move this one to the right
        manager.move_right()
        assert manager.left_of_current == (2,)
        assert manager.right_of_current == (1,)
        # Note that only the ngrams which are "in the game" are in the lists.
        assert manager.ngrams_left_side_of_current == [(2,)]
        assert manager.ngrams_right_side_of_current == [(1,)]
        manager.move_left()
        assert manager.left_of_current is None
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == [(2,)]

        # Test BACK functionality
        manager.move_back()
        assert manager.left_of_current == (2,)
        assert manager.right_of_current == (1,)
        # Note that only the ngrams which are "in the game" are in the lists.
        assert manager.ngrams_left_side_of_current == [(2,)]
        assert manager.ngrams_right_side_of_current == [(1,)]
        manager.move_right()
        assert manager.left_of_current == (1,)
        assert manager.right_of_current is None
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == []

        # Now let's start placing the (4,) again.
        manager.reset_current_ngram()
        assert manager.current_ngram == (4,)
        assert manager.ordered_ngrams == [(3,), (0,), (2,), (1,)]
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == [(3,), (0,)]
        assert manager.ngrams_right_side_of_current == [(2,), (1,)]

        # And let's rewind back to placing (2,)
        manager.previous_ngram()
        manager.previous_ngram()
        assert manager.ordered_ngrams == [(0,), (1,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (1,)

        # Let's then add these back, but in different order
        manager.move_left()
        manager.place_current_ngram()
        manager.move_right()
        manager.move_right()
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(2,), (0,), (1,), (3,)]
        assert manager.current_ngram == (4,)
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (1,)
        assert manager.ngrams_left_side_of_current == [(2,), (0,)]
        assert manager.ngrams_right_side_of_current == [(1,), (3,)]

        # Let's continue placing the rest
        manager.move_left()
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(2,), (4,), (0,), (1,), (3,)]
        assert manager.current_ngram == (5,)
        assert manager.left_of_current == (4,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(2,), (4,)]
        assert manager.ngrams_right_side_of_current == [(0,), (1,), (3,)]
        manager.move_right()

        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(2,), (4,), (0,), (1,), (5,), (3,)]
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(2,), (4,), (0,), (6,), (1,), (5,), (3,)]
        manager.place_current_ngram()  # calling extra times would NOT add the same item.
        assert manager.ordered_ngrams == [(2,), (4,), (0,), (6,), (1,), (5,), (3,)]

    def test_corner_cases_after_finished(self):
        manager = NgramPlacementManager(permutations=[(0,), (1,), (2,)])
        assert manager.all_ngrams == [(0,), (1,), (2,)]
        manager.place_current_ngram()
        manager.place_current_ngram()
        manager.place_current_ngram()
        assert manager.is_finished()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

        # Should not be possible to move anymore as the last item has been placed.
        manager.move_right()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]
        manager.move_left()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]
        manager.move_back()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

        # placing again should not do anything
        manager.place_current_ngram()
        assert manager.is_finished()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

        # resetting should not do anything
        manager.reset_current_ngram()
        assert manager.is_finished()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

        # it's possible to go back
        manager.previous_ngram()
        assert not manager.is_finished()
        assert manager.ordered_ngrams == [(1,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

        # ..and coming back will place the last item again.
        manager.place_current_ngram()
        assert manager.is_finished()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (2,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(0,)]

    def test_move_to_left_when_there_is_nothing_is_not_possible(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        assert manager.current_ngram == (0,)
        assert manager.ordered_ngrams == []

        manager.place_current_ngram()
        assert manager.left_of_current is None
        assert manager.current_ngram == (1,)
        assert manager.right_of_current == (0,)

        manager.move_left()
        assert manager.left_of_current is None
        assert manager.current_ngram == (1,)
        assert manager.right_of_current == (0,)

    def test_move_to_right_when_there_is_nothing_is_not_possible(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        assert manager.current_ngram == (0,)
        assert manager.ordered_ngrams == []

        manager.place_current_ngram()
        assert manager.left_of_current is None
        assert manager.current_ngram == (1,)
        assert manager.right_of_current == (0,)

        manager.move_right()
        assert manager.left_of_current == (0,)
        assert manager.current_ngram == (1,)
        assert manager.right_of_current is None

        # Moving again does not change anything.
        manager.move_right()
        assert manager.left_of_current == (0,)
        assert manager.current_ngram == (1,)
        assert manager.right_of_current is None

    def test_moving_back_too_many_times_is_okay(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        # Place some ngrams
        manager.place_current_ngram()
        manager.place_current_ngram()
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (3,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(2,), (0,)]

        # Create history of movements
        manager.move_left()
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == [(1,)]

        manager.move_right()
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == []

        # Calling move back many times should not crash
        manager.move_back()
        assert manager.ngrams_left_side_of_current == []
        assert manager.ngrams_right_side_of_current == [(1,)]
        manager.move_back()
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(2,), (0,)]

        # Some extra, just to test if it makes any difference.
        manager.move_back()
        manager.move_back()

        # Check that we are in the beginning state
        assert manager.ordered_ngrams == [(1,), (2,), (0,)]
        assert manager.current_ngram == (3,)
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (2,)
        assert manager.ngrams_left_side_of_current == [(1,)]
        assert manager.ngrams_right_side_of_current == [(2,), (0,)]

    def test_loading_state(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        manager.load_state([(0,), (3,), (1,), (2,)])
        assert manager.ordered_ngrams == [(0,), (3,), (1,), (2,)]
        assert manager.current_ngram == (4,)
        assert manager.left_of_current == (3,)
        assert manager.right_of_current == (1,)
        assert manager.ngrams_left_side_of_current == [(0,), (3,)]
        assert manager.ngrams_right_side_of_current == [(1,), (2,)]

        #  Adding new ngrams should continue from the last one
        manager.place_current_ngram()
        assert manager.ordered_ngrams == [(0,), (3,), (4,), (1,), (2,)]
        assert manager.current_ngram == (5,)
        assert manager.left_of_current == (3,)
        assert manager.right_of_current == (4,)

        # Try going backwards, even beyond the loaded state
        manager.previous_ngram()
        manager.previous_ngram()
        assert manager.ordered_ngrams == [(0,), (1,), (2,)]
        assert manager.current_ngram == (3,)
        assert manager.left_of_current == (0,)
        assert manager.right_of_current == (1,)
        assert manager.ngrams_left_side_of_current == [(0,)]
        assert manager.ngrams_right_side_of_current == [(1,), (2,)]

    def test_loading_bad_state(self):
        manager = NgramPlacementManager(permutations=self.permutations)
        assert manager.all_ngrams == [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        with pytest.raises(ValueError):
            # The 9999 is not in the permutations
            manager.load_state([(0,), (9999,), (1,), (2,)])

    def test_ordered_ngrams_area_widths(self):
        manager = NgramPlacementManager(
            permutations=[(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
        )
        manager.place_current_ngram()
        manager.place_current_ngram()
        manager.place_current_ngram()
        manager.place_current_ngram()

        assert manager.current_ngram == (4,)
        assert manager.ordered_ngrams == [(1,), (3,), (2,), (0,)]
        assert manager.left_of_current == (3,)
        assert manager.right_of_current == (2,)
        assert manager.ordered_ngrams_area_widths() == (0, 2, 2, 0)

        manager.move_left()
        assert manager.left_of_current == (1,)
        assert manager.right_of_current == (3,)
        assert manager.ordered_ngrams_area_widths() == (0, 1, 1, 2)

        manager.move_left()
        assert manager.left_of_current is None
        assert manager.right_of_current == (1,)
        assert manager.ordered_ngrams_area_widths() == (0, 0, 1, 3)

        manager.move_right()
        assert manager.left_of_current == (1,)
        assert manager.right_of_current is None
        assert manager.ordered_ngrams_area_widths() == (0, 1, 0, 3)

        manager.place_current_ngram()
        assert manager.current_ngram == (5,)
        assert manager.ordered_ngrams == [(1,), (4,), (3,), (2,), (0,)]
        assert manager.left_of_current == (4,)
        assert manager.right_of_current == (3,)
        assert manager.ngrams_left_side_of_current == [(1,), (4,)]
        assert manager.ngrams_right_side_of_current == [(3,), (2,), (0,)]
        assert manager.ordered_ngrams_area_widths() == (0, 2, 3, 0)

        manager.move_right()
        assert manager.left_of_current == (2,)
        assert manager.right_of_current == (0,)
        assert manager.ngrams_left_side_of_current == [(3,), (2,)]
        assert manager.ngrams_right_side_of_current == [(0,)]
        assert manager.ordered_ngrams_area_widths() == (2, 2, 1, 0)

        manager.move_right()
        assert manager.left_of_current == (0,)
        assert manager.right_of_current is None
        assert manager.ngrams_left_side_of_current == [(0,)]
        assert manager.ngrams_right_side_of_current == []
        assert manager.ordered_ngrams_area_widths() == (4, 1, 0, 0)


class TestSplittingOrderedNgrams:

    def test_even_number_of_seqs_right(self):
        ordered_seqs = [(0,), (1,), (2,), (3,), (4,), (5,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs, larger_size="right"
        )
        assert left_seqs == [(0,), (1,), (2,)]
        assert right_seqs == [(3,), (4,), (5,)]
        assert left == (2,)
        assert right == (3,)

    def test_even_number_of_seqs_left(self):
        ordered_seqs = [(0,), (1,), (2,), (3,), (4,), (5,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs, larger_size="left"
        )
        assert left_seqs == [(0,), (1,), (2,)]
        assert right_seqs == [(3,), (4,), (5,)]
        assert left == (2,)
        assert right == (3,)

    def test_odd_number_of_seqs_right(self):
        ordered_seqs = [(0,), (1,), (2,), (3,), (4,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs, larger_size="right"
        )
        assert left_seqs == [(0,), (1,)]
        assert right_seqs == [(2,), (3,), (4,)]
        assert left == (1,)
        assert right == (2,)

    def test_odd_number_of_seqs_left(self):
        ordered_seqs = [(0,), (1,), (2,), (3,), (4,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs, larger_size="left"
        )
        assert left_seqs == [(0,), (1,), (2,)]
        assert right_seqs == [(3,), (4,)]
        assert left == (2,)
        assert right == (3,)

    def test_single_seq(self):
        ordered_seqs = [(0,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs
        )
        assert left_seqs == []
        assert right_seqs == [(0,)]
        assert left is None
        assert right == (0,)

    def test_single_seq_larger_side_left(self):
        ordered_seqs = [(0,)]
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            ordered_seqs, larger_size="left"
        )
        assert left_seqs == [(0,)]
        assert right_seqs == []
        assert left == (0,)
        assert right is None

    def test_no_seqs(self):
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs([])
        assert left_seqs == []
        assert right_seqs == []
        assert left is None
        assert right is None

    def test_no_seqs_left(self):
        left, right, left_seqs, right_seqs = split_ordered_ngrams_into_two_halfs(
            [], larger_size="left"
        )
        assert left_seqs == []
        assert right_seqs == []
        assert left is None
        assert right is None
