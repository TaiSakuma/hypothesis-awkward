import math
from typing import Any

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.builtins_ import (
    builtin_safe_dtypes,
    from_list,
    items_from_dtype,
    lists,
)


@given(data=st.data())
def test_items_from_dtype(data: st.DataObject) -> None:
    dtype = data.draw(builtin_safe_dtypes(), label='dtype')
    item = data.draw(items_from_dtype(dtype), label='item')

    def _to_dtype_to_item(dtype: np.dtype, item: Any) -> Any:
        if dtype.kind == 'M':  # datetime64
            # datetime64 requires the unit.
            unit, _ = np.datetime_data(dtype)
            n = dtype.type(item, unit)
            assert not isinstance(item, int)
        else:
            n = dtype.type(item)
        return n.item()

    assert _to_dtype_to_item(dtype, item) == item


@given(data=st.data())
def test_lists(data: st.DataObject) -> None:
    # Draw options
    allow_nan = data.draw(st.booleans(), label='allow_nan')
    max_size = data.draw(st.integers(min_value=0, max_value=5), label='max_size')
    max_depth = data.draw(st.integers(min_value=1, max_value=5), label='max_depth')

    # Call the test subject
    l = data.draw(
        lists(allow_nan=allow_nan, max_size=max_size, max_depth=max_depth), label='l'
    )
    assert isinstance(l, list)

    # Assert the options were effective
    def _is_nan(x: Any) -> bool:
        if x is None:
            # `NaT` becomes `None`
            return True
        elif isinstance(x, complex):
            return math.isnan(x.real) or math.isnan(x.imag)
        elif isinstance(x, float):
            return math.isnan(x)
        return False

    def _examine(l: list) -> tuple[bool, int, int]:
        '''Return (has_nan, max_size, depth) of the list `l`.'''
        if not isinstance(l, list):
            return (_is_nan(l), 0, 0)
        has_nan = False
        size = len(l)  # max length of lists
        depth = 1
        for item in l:
            h, s, d = _examine(item)
            has_nan = has_nan or h
            size = max(size, s)
            depth = max(depth, d + 1)
        return (has_nan, size, depth)

    has_nan, size, depth = _examine(l)
    note(f'{has_nan=}, {size=}, {depth=}')

    if not allow_nan:
        assert not has_nan
    assert size <= max_size
    assert depth <= max_depth

    # Assert an Awkward Array can be created.
    a = ak.Array(l)
    note(f'{a=}')
    assert isinstance(a, ak.Array)

    to_list = a.to_list()
    note(f'{to_list=}')

    if not has_nan:
        assert to_list == l


@given(data=st.data())
def test_from_list(data: st.DataObject) -> None:
    # Draw options
    allow_nan = data.draw(st.booleans(), label='allow_nan')
    max_size = data.draw(st.integers(min_value=0, max_value=5), label='max_size')
    max_depth = data.draw(st.integers(min_value=1, max_value=5), label='max_depth')

    # Call the test subject
    a = data.draw(
        from_list(allow_nan=allow_nan, max_size=max_size, max_depth=max_depth),
        label='a',
    )
    assert isinstance(a, ak.Array)
