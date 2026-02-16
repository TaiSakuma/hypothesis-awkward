from typing import Any
from unittest.mock import Mock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.strategies.constructors import array_ as array_module
from tests.strategies.contents.test_content import ContentsKwargs, contents_kwargs


class ArraysKwargs(ContentsKwargs, total=False):
    '''Options for `arrays()` strategy.'''


DEFAULTS = ArraysKwargs(
    dtypes=None,
    max_size=10,
    allow_nan=False,
    allow_numpy=True,
    allow_empty=True,
    allow_string=True,
    allow_bytestring=True,
    allow_regular=True,
    allow_list_offset=True,
    allow_list=True,
    max_depth=5,
)


@st.composite
def arrays_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ArraysKwargs]:
    '''Strategy for options for `arrays()` strategy.'''
    return draw(contents_kwargs(chain=chain))


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` forwards kwargs to `contents()` and wraps in `ak.Array`.'''
    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()
    kwargs = opts.kwargs

    drawn_layout: ak.contents.Content | None = None
    raised_exc: Exception | None = None

    def tracking_contents(*a: Any, **kw: Any) -> st.SearchStrategy[ak.contents.Content]:
        nonlocal drawn_layout, raised_exc
        strategy = st_ak.contents.contents(*a, **kw)

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

    mock_st_ak = Mock(wraps=st_ak)
    mock_st_ak.contents.contents = Mock(side_effect=tracking_contents)

    with patch.object(array_module, 'st_ak', mock_st_ak):
        try:
            a = data.draw(st_ak.constructors.arrays(**kwargs), label='a')
        except Exception as e:
            assert raised_exc is not None
            assert e is raised_exc
        else:
            assert isinstance(a, ak.Array)
            assert a.layout is drawn_layout

    mock_st_ak.contents.contents.assert_called_once_with(**{**DEFAULTS, **kwargs})
