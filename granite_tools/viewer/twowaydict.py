from textual._two_way_dict import TwoWayDict
from textual.widgets.data_table import RowKey


def change_twowaydct_value(
    dct: TwoWayDict[RowKey, int], oldvalue: int, newvalue: int
) -> None:
    key = dct.get_key(oldvalue)
    if key is None:
        raise ValueError(f"Value {oldvalue} not found in the TwoWayDict")
    change_twowaydct_value_for_key(dct, key, newvalue)


def change_twowaydct_value_for_key(
    dct: TwoWayDict[RowKey, int], key: RowKey, newvalue: int
) -> None:
    if dct.get_key(newvalue) is not None:
        raise ValueError(f"Value {newvalue} is already in the TwoWayDict.")
    # The deletion is required as otherwise one of the two-way dicts will have
    # the old value floating around..
    del dct[key]
    dct[key] = newvalue
