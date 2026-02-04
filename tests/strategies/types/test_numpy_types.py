from typing import TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(data=st.data())
def test_record_draws(data: st.DataObject) -> None:
    '''Test that st_ak.RecordDraws records drawn values.'''
    recorder = st_ak.RecordDraws(st_ak.supported_dtypes())
    n = data.draw(st.integers(min_value=0, max_value=10), label='n')
    expected = []
    for i in range(n):
        expected.append(data.draw(recorder, label=f'{i}'))
    assert recorder.drawn == expected


class NumpyTypesKwargs(TypedDict, total=False):
    '''Options for `numpy_types()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_datetime: bool


def numpy_types_kwargs() -> st.SearchStrategy[NumpyTypesKwargs]:
    '''Strategy for options for `numpy_types()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtypes': st.one_of(
                st.none(),
                st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
            ),
            'allow_datetime': st.booleans(),
        },
    ).map(lambda d: cast(NumpyTypesKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_types(data: st.DataObject) -> None:
    '''Test that `numpy_types()` respects all its options.'''
    # Draw options
    kwargs = data.draw(numpy_types_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.numpy_types(**kwargs), label='result')

    # Assert the result is a NumpyType
    assert isinstance(result, ak.types.NumpyType)

    # Assert the options were effective
    dtypes = kwargs.get('dtypes', None)
    allow_datetime = kwargs.get('allow_datetime', True)

    match dtypes:
        case None:
            assert result.primitive in st_ak.numpy.dtype.SUPPORTED_DTYPE_NAMES
            if not allow_datetime:
                assert not result.primitive.startswith('datetime64')
                assert not result.primitive.startswith('timedelta64')
        case st_ak.RecordDraws():
            drawn_dtypes = {d.name for d in dtypes.drawn}
            assert result.primitive in drawn_dtypes


def test_draw_integer_type() -> None:
    '''Assert that integer NumpyType can be drawn.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive in ('int8', 'int16', 'int32', 'int64'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_unsigned_integer_type() -> None:
    '''Assert that unsigned integer NumpyType can be drawn.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive in ('uint8', 'uint16', 'uint32', 'uint64'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_float_type() -> None:
    '''Assert that floating point NumpyType can be drawn.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive in ('float16', 'float32', 'float64'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_complex_type() -> None:
    '''Assert that complex NumpyType can be drawn.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive in ('complex64', 'complex128'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_bool_type() -> None:
    '''Assert that bool NumpyType can be drawn.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive == 'bool',
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_datetime64_type() -> None:
    '''Assert that datetime64 NumpyType can be drawn by default.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive.startswith('datetime64'),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_timedelta64_type() -> None:
    '''Assert that timedelta64 NumpyType can be drawn by default.'''
    find(
        st_ak.numpy_types(),
        lambda t: t.primitive.startswith('timedelta64'),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
