from typing import TypedDict, cast

import numpy as np
from hypothesis import given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import any_nan_nat_in_awkward_array


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
    max_size = kwargs.get('max_size', 10)

    def _leaf_dtypes(a: ak.Array) -> set[np.dtype]:
        '''Dtypes of leaf NumPy arrays contained in `a`.'''
        dtypes = set()

        def _visitor(layout):
            match layout:
                case ak.contents.NumpyArray():
                    dtypes.add(layout.data.dtype)
                case ak.contents.RecordArray():
                    for c in layout.contents:
                        _visitor(c)
                case _:
                    raise TypeError(f'Unexpected type: {type(layout)}')

        _visitor(a.layout)
        return dtypes

    def _is_structured(a: ak.Array) -> bool:
        layout = a.layout
        if isinstance(layout, ak.contents.NumpyArray):  # simple array
            return False
        assert isinstance(layout, ak.contents.RecordArray)  # structured array
        return True

    def _size(a: ak.Array) -> int:
        ret = 0

        def _visitor(layout):
            nonlocal ret
            match layout:
                case ak.contents.NumpyArray():
                    ret += layout.data.size
                case ak.contents.RecordArray():
                    for c in layout.contents:
                        _visitor(c)
                case _:
                    raise TypeError(f'Unexpected type: {type(layout)}')

        _visitor(a.layout)
        return ret

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
