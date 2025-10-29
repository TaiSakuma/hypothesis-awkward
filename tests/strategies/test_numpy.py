from typing import TypedDict

import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


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


def _dtype_kinds(d: np.dtype) -> set[str]:
    '''Kinds of simple dtypes (e.g. `i`, `f`, `M`) contained in `d`.'''
    if d.names is None:  # simple dtype
        kind = d.kind
        if kind == 'V' and d.subdtype is not None:
            kind = d.subdtype[0].kind
        return {kind}
    else:  # structured dtype
        kinds = set()
        for name in d.names:
            f = d.fields
            assert f is not None
            kinds.update(_dtype_kinds(f[name][0]))
        return kinds


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


class NumpyArraysKwargs(TypedDict, total=False):
    '''Options for `numpy_arrays()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool


@st.composite
def numpy_arrays_kwargs(draw: st.DrawFn) -> NumpyArraysKwargs:
    '''Strategy for options to `numpy_arrays()` strategy.'''
    kwargs = NumpyArraysKwargs()

    if draw(st.booleans()):
        kwargs['dtype'] = draw(
            st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            )
        )

    if draw(st.booleans()):
        kwargs['allow_structured'] = draw(st.booleans())

    if draw(st.booleans()):
        kwargs['allow_nan'] = draw(st.booleans())

    return kwargs


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_arrays(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(numpy_arrays_kwargs(), label='kwargs')

    # Call the test subject
    n = data.draw(st_ak.numpy_arrays(**kwargs), label='n')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_structured = kwargs.get('allow_structured', True)
    allow_nan = kwargs.get('allow_nan', False)

    def _has_nan(n: np.ndarray) -> bool:
        kind = n.dtype.kind
        match kind:
            case 'V':  # structured
                return any(_has_nan(n[field]) for field in n.dtype.names)
            case 'f' | 'c':  # float or complex
                return bool(np.any(np.isnan(n)))
            case 'm' | 'M':  # timedelta or datetime
                return bool(np.any(np.isnat(n)))
            case _:
                return False

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        kinds = _dtype_kinds(n.dtype)
        assert len(kinds) == 1
        assert dtype.kind in kinds

    structured = n.dtype.names is not None
    has_nan = _has_nan(n)

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    # Assert an Awkward Array can be created.
    a = ak.from_numpy(n)
    note(f'{a=}')
    assert isinstance(a, ak.Array)

    # Test if the NumPy array and Awkward Array are converted to the same list.
    # Compare only when `NaN` isn't allowed.
    # Structured arrays are known to result in a different list sometimes.
    to_list = a.to_list()
    note(f'{to_list=}')

    if not allow_nan:
        if not structured:  # simple array
            assert to_list == n.tolist()
        else:  # structured array
            # assert to_list == n.tolist()  # NOTE: Fails sometimes
            pass

    # Test if the Awkward Array is converted back to a NumPy array with the identical
    # values. The conversion of structured arrays fails under a known condition.
    # Structured arrays may not result in identical values.

    def _is_numpy_convertible(a: ak.Array) -> bool:
        '''True if `a.to_numpy()` is expected to work without error.

        `to_numpy()` fails for structured arrays with non-1D fields
        https://github.com/scikit-hep/awkward/issues/3690


        '''
        layout = a.layout
        if isinstance(layout, ak.contents.NumpyArray):  # simple array
            return True
        assert isinstance(layout, ak.contents.RecordArray)  # structured array
        return all(len(c.shape) == 1 for c in layout.contents)

    if _is_numpy_convertible(a):
        to_numpy = a.to_numpy()
        note(f'{to_numpy=}')
        if not has_nan:
            if not structured:
                np.testing.assert_array_equal(to_numpy, n)
            else:
                # np.testing.assert_array_equal(to_numpy, n)  # NOTE: Fails sometimes
                pass
    else:
        with pytest.raises(ValueError):
            a.to_numpy()


class FromNumpyKwargs(TypedDict, total=False):
    '''Options for `from_numpy()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool


@st.composite
def from_numpy_kwargs(draw: st.DrawFn) -> FromNumpyKwargs:
    '''Strategy for options to `from_numpy()` strategy.'''
    kwargs = FromNumpyKwargs()

    if draw(st.booleans()):
        kwargs['dtype'] = draw(
            st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            )
        )

    if draw(st.booleans()):
        kwargs['allow_structured'] = draw(st.booleans())

    if draw(st.booleans()):
        kwargs['allow_nan'] = draw(st.booleans())

    return kwargs


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

    def _has_nan(
        a: ak.Array | ak.contents.RecordArray | ak.contents.NumpyArray,
    ) -> bool:
        match a:
            case ak.Array():
                return _has_nan(a.layout)
            case ak.contents.RecordArray():
                return any(_has_nan(a[field]) for field in a.fields)
            case ak.contents.NumpyArray():
                arr = a.data
                kind = arr.dtype.kind
                if kind in {'f', 'c'}:
                    return bool(np.any(np.isnan(arr)))
                elif kind in {'m', 'M'}:
                    return bool(np.any(np.isnat(arr)))
                else:
                    return False
            case _:
                raise TypeError(f'Unexpected type: {type(a)}')

    dtypes = _leaf_dtypes(a)
    structured = _is_structured(a)
    has_nan = _has_nan(a)
    note(f'{dtypes=}')
    note(f'{structured=}')
    note(f'{has_nan=}')

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        assert len(dtypes) == 1
        assert dtype in dtypes

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan
