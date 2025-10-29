from typing import TypedDict

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

from .util import _dtype_kinds


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


@st.composite
def numpy_dtypes_kwargs(draw: st.DrawFn) -> NumpyDtypesKwargs:
    '''Strategy for options to `numpy_dtypes()` strategy.'''
    kwargs = NumpyDtypesKwargs()

    if draw(st.booleans()):
        kwargs['dtype'] = draw(
            st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            )
        )

    if draw(st.booleans()):
        kwargs['allow_array'] = draw(st.booleans())

    return kwargs




@given(data=st.data())
def test_numpy_dtypes(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(numpy_dtypes_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.numpy_dtypes(**kwargs), label='dtype')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_array = kwargs.get('allow_array', True)

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        kinds = _dtype_kinds(result)
        assert len(kinds) == 1
        assert dtype.kind in kinds
    if not allow_array:
        assert result.names is None  # not structured

    # Assert an Awkward Array can be created.
    ak.from_numpy(np.array([], dtype=result))
