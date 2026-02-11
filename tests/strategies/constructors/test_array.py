from typing import TypedDict, cast
from unittest.mock import patch

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


class ArraysKwargs(TypedDict, total=False):
    '''Options for `arrays()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_numpy: bool
    allow_empty: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    max_depth: int


DEFAULTS: ArraysKwargs = {
    'dtypes': None,
    'max_size': 10,
    'allow_nan': False,
    'allow_numpy': True,
    'allow_empty': True,
    'allow_regular': True,
    'allow_list_offset': True,
    'allow_list': True,
    'max_depth': 5,
}


def arrays_kwargs() -> st.SearchStrategy[ArraysKwargs]:
    '''Strategy for options for `arrays()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtypes': st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
            ),
            'max_size': st.integers(min_value=0, max_value=50),
            'allow_nan': st.booleans(),
            'allow_numpy': st.booleans(),
            'allow_empty': st.booleans(),
            'allow_regular': st.booleans(),
            'allow_list_offset': st.booleans(),
            'allow_list': st.booleans(),
            'max_depth': st.integers(min_value=0, max_value=5),
        },
    ).map(lambda d: cast(ArraysKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` forwards kwargs to `contents()` and wraps in `ak.Array`.'''
    kwargs = data.draw(arrays_kwargs(), label='kwargs')
    sentinel = ak.contents.NumpyArray(np.array([1, 2, 3]))

    with patch.object(
        st_ak.contents, 'contents', return_value=st.just(sentinel)
    ) as mock:
        a = data.draw(st_ak.constructors.arrays(**kwargs), label='a')

    mock.assert_called_once_with(**{**DEFAULTS, **kwargs})
    assert isinstance(a, ak.Array)
    assert a.layout is sentinel
