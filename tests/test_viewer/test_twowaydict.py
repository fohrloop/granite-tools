import pytest
from textual._two_way_dict import TwoWayDict
from textual.widgets.data_table import RowKey

from granite_tools.viewer.twowaydict import change_twowaydct_value


class TestTwoWayDict:

    def test_simple(self):
        dct = TwoWayDict(
            {
                RowKey("foo"): 1,
                RowKey("bar"): 2,
            }
        )

        assert len(dct) == 2

        # Save the keys (these do not change)
        key1 = dct.get_key(1)
        key2 = dct.get_key(2)

        # Change the values
        change_twowaydct_value(dct, 1, 3)
        change_twowaydct_value(dct, 2, 4)
        # The values were 1 and 2, now they are 3 and 4
        assert dct.get(key1) == 3
        assert dct.get(key2) == 4

        # sanity check: the length of the two-way dict as well as the two internal dicts
        # are unchanged.
        assert len(dct) == 2
        assert len(dct._forward) == 2
        assert len(dct._reverse) == 2

    def test_with_three_items(self):
        dct_orig = {
            RowKey("foo"): 1,
            RowKey("bar"): 2,
            RowKey("baz"): 3,
        }
        dct = TwoWayDict(dct_orig)
        change_twowaydct_value(dct, 3, 5)

        assert list(dct._forward.values()) == [1, 2, 5]
        assert list(dct._reverse.keys()) == [1, 2, 5]

        assert dct.get_key(1).value == "foo"
        assert dct.get_key(2).value == "bar"
        assert dct.get_key(5).value == "baz"
        assert dct.get(RowKey("foo")) == 1
        assert dct.get(RowKey("bar")) == 2
        assert dct.get(RowKey("baz")) == 5

    def test_not_possible_to_make_bad_dict(self):
        dct_orig = {
            RowKey("foo"): 1,
            RowKey("bar"): 2,
            RowKey("baz"): 3,
        }
        dct = TwoWayDict(dct_orig)
        with pytest.raises(ValueError):
            # This would create two two's (2)
            change_twowaydct_value(dct, 3, 2)
