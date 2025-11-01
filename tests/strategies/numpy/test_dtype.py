from typing import TypedDict, cast

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import n_scalars_in, simple_dtype_kinds_in


@given(name=st_ak.supported_dtype_names())
def test_supported_dtype_names(name: str) -> None:
    ak.from_numpy(np.array([], dtype=name))


@given(dtype=st_ak.supported_dtypes())
def test_supported_dtypes(dtype: np.dtype) -> None:
    ak.from_numpy(np.array([], dtype=dtype))


class NumpyDtypesKwargs(TypedDict, total=False):
    '''Options for `numpy_dtypes()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_array: bool
    max_size: int


def numpy_dtypes_kwargs() -> st.SearchStrategy[NumpyDtypesKwargs]:
    '''Strategy for options for `numpy_dtypes()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            ),
            'allow_array': st.booleans(),
            'max_size': st.integers(min_value=1, max_value=10),
        },
    ).map(lambda d: cast(NumpyDtypesKwargs, d))


@given(data=st.data())
def test_numpy_dtypes(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(numpy_dtypes_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.numpy_dtypes(**kwargs), label='dtype')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_array = kwargs.get('allow_array', True)
    max_size = kwargs.get('max_size', 5)

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        kinds = simple_dtype_kinds_in(result)
        assert len(kinds) == 1
        assert dtype.kind in kinds
    if not allow_array:
        assert result.names is None  # not structured
    if max_size is not None:
        assert n_scalars_in(result) <= max_size

    # Assert an Awkward Array can be created.
    ak.from_numpy(np.array([], dtype=result))
