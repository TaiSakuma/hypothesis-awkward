import math
from typing import Any, TypedDict

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(data=st.data())
def test_items_from_dtype(data: st.DataObject) -> None:
    dtype = data.draw(st_ak.builtin_safe_dtypes(), label='dtype')
    item = data.draw(st_ak.items_from_dtype(dtype), label='item')

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


class ListsKwargs(TypedDict, total=False):
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_size: int
    max_depth: int


@st.composite
def lists_kwargs(draw: st.DrawFn) -> ListsKwargs:
    kwargs = ListsKwargs()

    if draw(st.booleans()):
        kwargs['dtype'] = draw(
            st.one_of(
                st.none(),
                st.just(st_ak.builtin_safe_dtypes()),
                st_ak.builtin_safe_dtypes(),
            )
        )

    if draw(st.booleans()):
        kwargs['allow_nan'] = draw(st.booleans())

    if draw(st.booleans()):
        kwargs['max_size'] = draw(st.integers(min_value=0, max_value=5))

    if draw(st.booleans()):
        kwargs['max_depth'] = draw(st.integers(min_value=1, max_value=5))

    return kwargs


@given(data=st.data())
def test_lists(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(lists_kwargs(), label='kwargs')

    # Call the test subject
    l = data.draw(st_ak.lists(**kwargs), label='l')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_nan = kwargs.get('allow_nan', False)
    max_size = kwargs.get('max_size', 5)
    max_depth = kwargs.get('max_depth', 5)

    def _is_nan(x: Any) -> bool:
        if x is None:
            # `NaT` becomes `None`
            return True
        elif isinstance(x, complex):
            return math.isnan(x.real) or math.isnan(x.imag)
        elif isinstance(x, float):
            return math.isnan(x)
        return False

    def _examine(l: list) -> tuple[bool, int, int, set[type]]:
        '''Return (has_nan, max_size, depth, types) of the list `l`.'''
        if not isinstance(l, list):
            is_nan = _is_nan(l)
            type_l = {type(l)} if not is_nan else set()
            return (is_nan, 0, 0, type_l)
        has_nan = False
        size = len(l)  # max length of lists
        depth = 1
        types = set()
        for item in l:
            h, s, d, t = _examine(item)
            has_nan = has_nan or h
            size = max(size, s)
            depth = max(depth, d + 1)
            types.update(t)
        return (has_nan, size, depth, types)

    has_nan, size, depth, types = _examine(l)
    note(f'{has_nan=}, {size=}, {depth=}, {types=}')

    assert len(types) <= 1  # All same type unless empty

    if isinstance(dtype, np.dtype) and types:
        type_ = next(iter(types))
        note(f'{type_=})')
        # TODO: Assert `dtype` matches `type_`

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
    kwargs = data.draw(lists_kwargs(), label='kwargs')

    # Call the test subject
    a = data.draw(st_ak.from_list(**kwargs), label='a')
    assert isinstance(a, ak.Array)
