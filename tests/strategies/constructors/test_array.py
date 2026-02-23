from typing import Any, cast
from unittest.mock import Mock, patch

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, UnionArray
from hypothesis_awkward.util import iter_contents
from tests.strategies.contents.test_content import ContentsKwargs, contents_kwargs


class ArraysKwargs(ContentsKwargs, total=False):
    '''Options for `arrays()` strategy.'''

    allow_virtual: bool


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
    allow_record=True,
    allow_union=True,
    max_depth=5,
    max_length=None,
    allow_virtual=True,
)


@st.composite
def arrays_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ArraysKwargs]:
    '''Strategy for options for `arrays()` strategy.'''
    opts = draw(contents_kwargs(chain=chain))
    extra = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'allow_virtual': st.booleans(),
            },
        )
    )
    return opts.extend(cast(ArraysKwargs, extra))


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` forwards kwargs to `contents()` and wraps in `ak.Array`.'''
    from hypothesis_awkward.strategies.constructors import array_ as array_module

    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()
    kwargs = opts.kwargs

    spy = ContentsSpy()
    mock_st_ak = Mock(wraps=st_ak)
    mock_st_ak.contents.contents = Mock(side_effect=spy.wrap_contents)

    contents_keys = ContentsKwargs.__required_keys__ | ContentsKwargs.__optional_keys__
    expected_contents_kwargs = {
        k: v for k, v in {**DEFAULTS, **kwargs}.items() if k in contents_keys
    }

    with patch.object(array_module, 'st_ak', mock_st_ak):
        try:
            a = data.draw(st_ak.constructors.arrays(**kwargs), label='a')
        except Exception as e:
            assert spy.raised_exc is not None
            assert e is spy.raised_exc
            return
        finally:
            mock_st_ak.contents.contents.assert_called_once_with(
                **expected_contents_kwargs
            )

    assert isinstance(a, ak.Array)
    is_virtual = not ak.to_layout(a).is_all_materialized
    allow_virtual = kwargs.get('allow_virtual', DEFAULTS['allow_virtual'])
    if not allow_virtual:
        assert not is_virtual
    if not is_virtual:
        assert a.layout is spy.drawn_layout
    else:
        assert spy.drawn_layout is not None
        assert a.layout.form == spy.drawn_layout.form

        assert_all_buffers_virtual(a)


def assert_all_buffers_virtual(a: ak.Array) -> None:
    '''Assert that all buffers that can remain virtual are not materialized.

    Some buffers may not have ``is_materialized``. We skip those and
    verify the rest.
    '''
    _form, _length, buffers = ak.to_buffers(a)
    for v in buffers.values():
        if hasattr(v, 'is_materialized'):
            assert not v.is_materialized


class ContentsSpy:
    '''Wraps `contents()` to capture the drawn layout and any raised exception.'''

    def __init__(self) -> None:
        self.drawn_layout: Content | None = None
        self.raised_exc: Exception | None = None

    def wrap_contents(self, *a: Any, **kw: Any) -> st.SearchStrategy[Content]:
        strategy = st_ak.contents.contents(*a, **kw)
        spy = self

        @st.composite
        def _draw(draw_inner: st.DrawFn) -> Content:
            spy.drawn_layout = None
            spy.raised_exc = None
            try:
                spy.drawn_layout = draw_inner(strategy)
            except Exception as e:
                spy.raised_exc = e
                raise
            return spy.drawn_layout

        return _draw()


def test_draw_virtual() -> None:
    '''Assert that virtual arrays can be drawn by default.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: not ak.to_layout(a).is_all_materialized,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_virtual_union() -> None:
    '''Assert that a virtual array containing a UnionArray can be drawn.'''

    def _is_virtual_union(a: ak.Array) -> bool:
        layout = ak.to_layout(a)
        has_union = any(isinstance(n, UnionArray) for n in iter_contents(layout))
        return has_union and not layout.is_all_materialized

    find(
        st_ak.constructors.arrays(max_size=20),
        _is_virtual_union,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
