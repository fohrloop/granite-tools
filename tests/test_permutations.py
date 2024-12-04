from granite_tools.hands import Hand
from granite_tools.permutations import (
    create_permutations,
    get_union_of_keys,
    permutation_is_typable,
)


class TestCreatePermutations:
    left = Hand(hand="Left", symbols_visualization={0: "x", 1: "y", 2: "z"})
    right = Hand(hand="Right", symbols_visualization={0: "a", 1: "b", 2: "c", 3: "d"})

    def test_simple(self):

        permutations = create_permutations(self.left, self.right, sequence_lengths=(1,))
        assert permutations == [(0,), (1,), (2,), (3,)]

    def test_2key(self):
        permutations = create_permutations(self.left, self.right, sequence_lengths=(2,))
        assert permutations == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
        ]

    def test_3key(self):
        permutations = create_permutations(
            left=Hand(hand="Left", symbols_visualization={0: "x", 1: "y"}),
            right=Hand(hand="Right", symbols_visualization={3: "d"}),
            sequence_lengths=(3,),
        )
        # In this case, the left hand can type all permutations with  0 and 1,
        # and right hand can only type (3,3,3)
        assert permutations == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (3, 3, 3),
        ]


class TestPermutationIsTypable:
    def test_is_typable(self):

        left = Hand(hand="Left", symbols_visualization={0: "x", 1: "y", 2: "z"})
        right = Hand(hand="Right", symbols_visualization={0: "x", 1: "y", 6: "z"})

        # Can be typed only with left
        assert permutation_is_typable(left, right, (2,)) == True
        # Can be typed only with right
        assert permutation_is_typable(left, right, (6,)) == True
        # Can be typed with both
        assert permutation_is_typable(left, right, (0, 1)) == True
        # Can be typed with left
        assert permutation_is_typable(left, right, (0, 1, 2)) == True
        # Can be typed with right
        assert permutation_is_typable(left, right, (0, 1, 6)) == True
        # Can't be typed with any hands (2 and 6 do exist, but in different hands)
        assert permutation_is_typable(left, right, (2, 6)) == False
        assert permutation_is_typable(left, right, (1, 2, 6)) == False


class TestCreateUnionOfKeys:

    def test_simple(self):
        left = Hand(hand="Left", symbols_visualization={0: "1", 1: "2", 2: "3"})
        right = Hand(hand="Right", symbols_visualization={0: "A", 1: "B"})

        assert get_union_of_keys(left, right) == [0, 1, 2]

    def test_simple_another(self):
        left = Hand(hand="Left", symbols_visualization={0: "1", 6: "6"})
        right = Hand(hand="Right", symbols_visualization={1: "B"})

        assert get_union_of_keys(left, right) == [0, 1, 6]
