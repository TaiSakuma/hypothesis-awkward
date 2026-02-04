from typing import TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.strategies.numpy.dtype import SUPPORTED_DTYPE_NAMES


class NumpyFormsKwargs(TypedDict, total=False):
    '''Options for `numpy_forms()` strategy.'''

    type_: ak.types.NumpyType | st.SearchStrategy[ak.types.NumpyType] | None
    dtypes: st.SearchStrategy[np.dtype] | None
    allow_datetime: bool
    inner_shape: tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] | None
    allow_inner_shape: bool


class NumpyFormsOpts:
    '''Drawn options for `numpy_forms()` with resettable recorders.'''

    def __init__(self, kwargs: NumpyFormsKwargs) -> None:
        self._kwargs = kwargs

    @property
    def kwargs(self) -> NumpyFormsKwargs:
        return self._kwargs

    def reset(self) -> None:
        for v in self._kwargs.values():
            if isinstance(v, st_ak.RecordDraws):
                v.drawn.clear()


def _inner_shape_strategies() -> st.SearchStrategy[tuple[int, ...]]:
    '''Strategy for generating small inner_shape tuples.'''
    return st.lists(
        st.integers(min_value=1, max_value=5),
        min_size=1,
        max_size=3,
    ).map(tuple)


def numpy_forms_kwargs() -> st.SearchStrategy[NumpyFormsOpts]:
    '''Strategy for options for `numpy_forms()` strategy.

    Two modes:
    - type_ mode: type_ is set, other params omitted.
    - dtypes mode: type_ omitted, dtypes/allow_datetime/inner_shape/allow_inner_shape drawn.
    '''

    def type_mode() -> st.SearchStrategy[NumpyFormsKwargs]:
        return st.fixed_dictionaries(
            {
                'type_': st.one_of(
                    st_ak.numpy_types(),
                    st.just(st_ak.RecordDraws(st_ak.numpy_types())),
                ),
            },
        ).map(lambda d: cast(NumpyFormsKwargs, d))

    def dtypes_mode() -> st.SearchStrategy[NumpyFormsKwargs]:
        return st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_datetime': st.booleans(),
                'inner_shape': st.one_of(
                    st.none(),
                    st.lists(
                        st.integers(min_value=1, max_value=5),
                        min_size=0,
                        max_size=3,
                    ).map(tuple),
                    st.just(st_ak.RecordDraws(_inner_shape_strategies())),
                ),
                'allow_inner_shape': st.booleans(),
            },
        ).map(lambda d: cast(NumpyFormsKwargs, d))

    return st.one_of(type_mode(), dtypes_mode()).map(NumpyFormsOpts)


DATETIME_PRIMITIVES = frozenset(
    n for n in SUPPORTED_DTYPE_NAMES if n.startswith(('datetime64', 'timedelta64'))
)


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_forms(data: st.DataObject) -> None:
    '''Test that `numpy_forms()` respects all its options.'''
    # Draw options
    opts = data.draw(numpy_forms_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(st_ak.numpy_forms(**opts.kwargs), label='result')

    # Assert the result is always a NumpyForm
    assert isinstance(result, ak.forms.NumpyForm)

    # Assert parameters and form_key are always None
    assert result._parameters is None
    assert result._form_key is None

    note(f'{result.primitive=}')
    note(f'{result.inner_shape=}')

    # Assert the options were effective
    type_ = opts.kwargs.get('type_', None)
    dtypes = opts.kwargs.get('dtypes', None)
    allow_datetime = opts.kwargs.get('allow_datetime', True)
    inner_shape = opts.kwargs.get('inner_shape', None)
    allow_inner_shape = opts.kwargs.get('allow_inner_shape', True)

    if type_ is not None:
        # type_ mode: primitive matches type, inner_shape is ()
        match type_:
            case ak.types.NumpyType():
                assert result.primitive == type_.primitive
            case st_ak.RecordDraws():
                drawn_primitives = {t.primitive for t in type_.drawn}
                assert result.primitive in drawn_primitives
        assert result.inner_shape == ()
    else:
        # dtypes mode
        match dtypes:
            case None:
                assert result.primitive in SUPPORTED_DTYPE_NAMES
                if not allow_datetime:
                    assert result.primitive not in DATETIME_PRIMITIVES
            case st_ak.RecordDraws():
                drawn_dtype_names = {d.name for d in dtypes.drawn}
                assert result.primitive in drawn_dtype_names

        match inner_shape:
            case None:
                if not allow_inner_shape:
                    assert result.inner_shape == ()
            case st_ak.RecordDraws():
                drawn_shapes = set(inner_shape.drawn)
                assert result.inner_shape in drawn_shapes
            case tuple():
                assert result.inner_shape == inner_shape


def test_draw_empty_inner_shape() -> None:
    '''Assert that forms with empty inner_shape can be drawn.'''
    find(
        st_ak.numpy_forms(),
        lambda f: f.inner_shape == (),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nonempty_inner_shape() -> None:
    '''Assert that forms with non-empty inner_shape can be drawn.'''
    find(
        st_ak.numpy_forms(),
        lambda f: len(f.inner_shape) > 0,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_datetime_primitive() -> None:
    '''Assert that datetime64 primitive can be drawn.'''
    find(
        st_ak.numpy_forms(),
        lambda f: f.primitive.startswith('datetime64'),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_integer_primitive() -> None:
    '''Assert that integer primitives can be drawn.'''
    find(
        st_ak.numpy_forms(),
        lambda f: f.primitive in ('int8', 'int16', 'int32', 'int64'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_from_type() -> None:
    '''Assert that a form from a NumpyType matches the type.'''
    t = ak.types.NumpyType('float64')
    f = find(
        st_ak.numpy_forms(type_=t),
        lambda f: True,
        settings=settings(phases=[Phase.generate]),
    )
    assert f.primitive == 'float64'
    assert f.inner_shape == ()
