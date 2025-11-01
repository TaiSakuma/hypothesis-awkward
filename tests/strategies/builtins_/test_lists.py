import math
from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


class ListsKwargs(TypedDict, total=False):
    '''Options for `lists()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_size: int


def lists_kwargs() -> st.SearchStrategy[ListsKwargs]:
    '''Strategy for options for `lists()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(
                st.none(),
                st.just(st_ak.builtin_safe_dtypes()),
                st_ak.builtin_safe_dtypes(),
            ),
            'allow_nan': st.booleans(),
            'max_size': st.integers(min_value=0, max_value=25),
        },
    ).map(lambda d: cast(ListsKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_lists(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(lists_kwargs(), label='kwargs')

    # Call the test subject
    l = data.draw(st_ak.lists(**kwargs), label='l')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_nan = kwargs.get('allow_nan', False)
    max_size = kwargs.get('max_size', 10)

    def _is_nan(x: Any) -> bool:
        if x is None:
            # `NaT` becomes `None`
            return True
        elif isinstance(x, complex):
            return math.isnan(x.real) or math.isnan(x.imag)
        elif isinstance(x, float):
            return math.isnan(x)
        return False

    def _flatten(l: list) -> Any:
        for i in l:
            if isinstance(i, list):
                yield from _flatten(i)
            else:
                yield i

    flat = list(_flatten(l))
    has_nan = any(_is_nan(x) for x in flat)
    size = len(flat)
    types = {type(x) for x in flat if not _is_nan(x)}
    note(f'{has_nan=}, {max_size=}, {types=}')

    assert len(types) <= 1  # All same type unless empty

    if isinstance(dtype, np.dtype) and types:
        type_ = next(iter(types))
        note(f'{type_=})')
        # TODO: Assert `dtype` matches `type_`

    if not allow_nan:
        assert not has_nan
    assert size <= max_size

    # Assert an Awkward Array can be created.
    a = ak.Array(l)
    note(f'{a=}')
    assert isinstance(a, ak.Array)

    to_list = a.to_list()
    note(f'{to_list=}')

    if not has_nan:
        assert to_list == l
