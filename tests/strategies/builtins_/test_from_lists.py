from typing import TypedDict, cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import any_nan_nat_in_numpy_array


class FromListsKwargs(TypedDict, total=False):
    '''Options for `from_list()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_size: int


def from_lists_kwargs() -> st.SearchStrategy[FromListsKwargs]:
    '''Strategy for options for `from_list()` strategy.'''
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
    ).map(lambda d: cast(FromListsKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_from_list(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(from_lists_kwargs(), label='kwargs')

    # Call the test subject
    a = data.draw(st_ak.from_list(**kwargs), label='a')
    assert isinstance(a, ak.Array)

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_nan = kwargs.get('allow_nan', False)
    max_size = kwargs.get('max_size', 10)

    n_flat = ak.flatten(a, axis=None).to_numpy()

    types = {type(x) for x in n_flat}
    assert len(types) <= 1  # All same type unless empty
    if isinstance(dtype, np.dtype) and types:
        assert dtype.type in types

    has_nan = any_nan_nat_in_numpy_array(n_flat)
    if not allow_nan:
        assert not has_nan

    assert len(n_flat) <= max_size
