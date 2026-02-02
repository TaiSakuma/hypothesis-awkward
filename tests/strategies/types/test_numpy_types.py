from typing import Any, TypedDict, TypeVar, cast

import numpy as np
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

T = TypeVar('T')


class RecordDraws(st.SearchStrategy[T]):
    '''Wrap a strategy to store a set of all drawn values.'''

    def __init__(self, base: st.SearchStrategy[T]) -> None:
        super().__init__()
        self.drawn = set[T]()
        self._base = base

    def do_draw(self, data: Any) -> T:
        value = data.draw(self._base)
        self.drawn.add(value)
        return value


@given(data=st.data())
def test_record_draws(data: st.DataObject) -> None:
    '''Test that RecordDraws records drawn values.'''
    recorder = RecordDraws(st_ak.supported_dtypes())
    n = data.draw(st.integers(min_value=0, max_value=10), label='n')
    expected = set()
    for i in range(n):
        expected.add(data.draw(recorder, label=f'{i}'))
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
                st.just(st_ak.supported_dtypes()),
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

    note(f'{result=}')
    note(f'{result.primitive=}')

    # If dtypes is None and allow_datetime is False, datetime types should not appear
    if dtypes is None and not allow_datetime:
        assert result.primitive not in ('datetime64', 'timedelta64')
        # Also check that it's not a datetime with units
        assert not result.primitive.startswith('datetime64')
        assert not result.primitive.startswith('timedelta64')


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
