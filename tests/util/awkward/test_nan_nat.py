import math
from collections.abc import Iterator

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import any_nan_nat_in_awkward_array


def st_arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = True,
    max_size: int = 10,
) -> st.SearchStrategy[ak.Array]:
    '''Tentative strategy for Awkward Arrays (combines from_numpy and from_list).'''
    return st.one_of(
        st_ak.from_numpy(dtype=dtype, allow_nan=allow_nan, max_size=max_size),
        # st_ak.from_list(dtype=dtype, allow_nan=allow_nan, max_size=max_size),
    )


@given(data=st.data())
def test_any_nan_nat_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_arrays(allow_nan=allow_nan))
    actual = any_nan_nat_in_awkward_array(a)
    if allow_nan:
        expected = _has_nan_nat_via_iteration(a)
    else:
        expected = False
    assert actual == expected


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn.'''
    find(
        st_arrays(),
        _has_nan_via_iteration,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn.'''
    find(
        st_arrays(),
        _has_nat_via_iteration,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def _has_nan_nat_via_iteration(a: ak.Array) -> bool:
    '''Check if array contains any NaN or NaT.'''
    return _has_nan_via_iteration(a) or _has_nat_via_iteration(a)


def _has_nan_via_iteration(a: ak.Array) -> bool:
    '''Check for NaN by iterating over underlying numpy arrays.'''
    for arr in _iter_numpy_arrays(a.layout):
        if arr.dtype.kind in {'f', 'c'}:
            for val in arr.flat:
                if isinstance(val, (complex, np.complexfloating)):
                    if math.isnan(val.real) or math.isnan(val.imag):
                        return True
                elif isinstance(val, (float, np.floating)):
                    if math.isnan(val):
                        return True
    return False


def _has_nat_via_iteration(a: ak.Array) -> bool:
    '''Check for NaT by iterating over underlying numpy arrays.'''
    for arr in _iter_numpy_arrays(a.layout):
        if arr.dtype.kind in {'m', 'M'}:
            for val in arr.flat:
                if np.isnat(val):
                    return True
    return False


def _iter_numpy_arrays(
    layout: ak.contents.Content,
) -> Iterator[np.ndarray]:
    '''Iterate over all numpy arrays in an awkward layout.'''
    stack: list[ak.contents.Content] = [layout]
    while stack:
        content = stack.pop()
        if isinstance(content, ak.contents.NumpyArray):
            yield content.data
        elif isinstance(content, ak.contents.RecordArray):
            for field in content.fields:
                stack.append(content[field])
