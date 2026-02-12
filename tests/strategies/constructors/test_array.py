from typing import Any, cast
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from tests.strategies.contents.test_content import ContentsKwargs


class ArraysKwargs(ContentsKwargs, total=False):
    '''Options for `arrays()` strategy.'''


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

    real_contents = st_ak.contents.contents
    drawn_layout: ak.contents.Content | None = None
    raised_exc: Exception | None = None

    def tracking_contents(*a: Any, **kw: Any) -> st.SearchStrategy[ak.contents.Content]:
        nonlocal drawn_layout, raised_exc
        strategy = real_contents(*a, **kw)

        @st.composite
        def wrapped(draw_inner: st.DrawFn) -> ak.contents.Content:
            nonlocal drawn_layout, raised_exc
            try:
                content = draw_inner(strategy)
            except Exception as e:
                raised_exc = e
                raise
            drawn_layout = content
            return content

        return wrapped()

    with patch.object(
        st_ak.contents, 'contents', side_effect=tracking_contents
    ) as mock:
        try:
            a = data.draw(st_ak.constructors.arrays(**kwargs), label='a')
        except Exception as e:
            assert raised_exc is not None
            assert e is raised_exc
        else:
            assert isinstance(a, ak.Array)
            assert a.layout is drawn_layout

    mock.assert_called_once_with(**{**DEFAULTS, **kwargs})
