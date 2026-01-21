from typing import TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
    any_nat_in_awkward_array,
    iter_numpy_arrays,
)

DEFAULT_MAX_SIZE = 10


class FromNumpyKwargs(TypedDict, total=False):
    '''Options for `from_numpy()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    max_size: int


def from_numpy_kwargs() -> st.SearchStrategy[FromNumpyKwargs]:
    '''Strategy for options for `from_numpy()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            ),
            'allow_structured': st.booleans(),
            'allow_nan': st.booleans(),
            'max_size': st.integers(min_value=0, max_value=100),
        },
    ).map(lambda d: cast(FromNumpyKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_from_numpy(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(from_numpy_kwargs(), label='kwargs')

    # Call the test subject
    a = data.draw(st_ak.from_numpy(**kwargs), label='a')
    assert isinstance(a, ak.Array)

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_structured = kwargs.get('allow_structured', True)
    allow_nan = kwargs.get('allow_nan', False)
    max_size = kwargs.get('max_size', DEFAULT_MAX_SIZE)

    dtypes = _leaf_dtypes(a)
    structured = _is_structured(a)
    has_nan = any_nan_nat_in_awkward_array(a)
    size = _size(a)
    note(f'{dtypes=}')
    note(f'{structured=}')
    note(f'{has_nan=}')
    note(f'{size=}')

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        assert len(dtypes) == 1
        assert dtype in dtypes

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    assert size <= max_size


def test_draw_structured() -> None:
    '''Assert that structured arrays can be drawn by default.'''
    find(
        st_ak.from_numpy(),
        lambda a: isinstance(a.layout, ak.contents.RecordArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    floating_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.from_numpy(dtype=floating_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat_datetime64() -> None:
    '''Assert that datetime64 arrays with NaT can be drawn when allowed.'''
    datetime64_dtypes = st_ak.supported_dtypes().filter(
        lambda d: d.kind == 'M'
    )
    find(
        st_ak.from_numpy(dtype=datetime64_dtypes, allow_nan=True),
        any_nat_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat_timedelta64() -> None:
    '''Assert that timedelta64 arrays with NaT can be drawn when allowed.'''
    timedelta64_dtypes = st_ak.supported_dtypes().filter(
        lambda d: d.kind == 'm'
    )
    find(
        st_ak.from_numpy(dtype=timedelta64_dtypes, allow_nan=True),
        any_nat_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.from_numpy(),
        lambda a: len(a) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with max_size elements can be drawn by default.'''
    find(
        st_ak.from_numpy(allow_structured=False),
        lambda a: _size(a) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def _leaf_dtypes(a: ak.Array) -> set[np.dtype]:
    '''Dtypes of leaf NumPy arrays contained in `a`.'''
    return {arr.dtype for arr in iter_numpy_arrays(a)}


def _is_structured(a: ak.Array) -> bool:
    '''Check if `a` is a structured array.'''
    layout = a.layout
    if isinstance(layout, ak.contents.NumpyArray):  # simple array
        return False
    assert isinstance(layout, ak.contents.RecordArray)  # structured array
    return True


def _size(a: ak.Array) -> int:
    '''Total size of all leaf NumPy arrays contained in `a`.'''
    return sum(arr.size for arr in iter_numpy_arrays(a))
