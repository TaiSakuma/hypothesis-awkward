from functools import partial
from typing import Generic, Optional, TypedDict, TypeVar

from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies import StMinMaxValuesFactory, none_or, ranges
from hypothesis_awkward.util import safe_compare as sc
from hypothesis_awkward.util import safe_max

T = TypeVar('T')


def min_max_starts(
    st_: StMinMaxValuesFactory[T],
) -> st.SearchStrategy[tuple[Optional[T], Optional[T]]]:
    def mins() -> st.SearchStrategy[Optional[T]]:
        return none_or(st_())

    def maxes(min_value: Optional[T]) -> st.SearchStrategy[Optional[T]]:
        return none_or(st_(min_value=min_value))

    return mins().flatmap(lambda min_: st.tuples(st.just(min_), maxes(min_)))


def min_max_ends(
    st_: StMinMaxValuesFactory[T],
    min_start: Optional[T] = None,
) -> st.SearchStrategy[tuple[Optional[T], Optional[T]]]:
    def mins() -> st.SearchStrategy[Optional[T]]:
        return none_or(st_(min_value=min_start))

    def maxes(min_value: Optional[T]) -> st.SearchStrategy[Optional[T]]:
        min_value = safe_max((min_value, min_start))
        return none_or(st_(min_value=min_value))

    return mins().flatmap(lambda min_: st.tuples(st.just(min_), maxes(min_)))


class RangesKwargs(TypedDict, Generic[T], total=False):
    # st_: StMinMaxValuesFactory[T]
    min_start: Optional[T]
    max_start: Optional[T]
    min_end: Optional[T]
    max_end: Optional[T]
    allow_start_none: bool
    allow_end_none: bool
    let_end_none_if_start_none: bool
    allow_equal: bool


@st.composite
def ranges_kwargs(
    draw: st.DrawFn, st_: StMinMaxValuesFactory[T] | None = None
) -> RangesKwargs[T]:
    st_ = st_ or st.integers  # type: ignore
    kwargs = RangesKwargs[T]()

    min_start, max_start = draw(min_max_starts(st_=st_))  # type: ignore
    if min_start is not None:
        kwargs['min_start'] = min_start
    if max_start is not None:
        kwargs['max_start'] = max_start

    min_end, max_end = draw(min_max_ends(st_=st_, min_start=min_start))  # type: ignore
    if min_end is not None:
        kwargs['min_end'] = min_end
    if max_end is not None:
        kwargs['max_end'] = max_end

    if draw(st.booleans()):
        kwargs['allow_start_none'] = draw(st.booleans())
    if draw(st.booleans()):
        kwargs['allow_end_none'] = draw(st.booleans())
    if draw(st.booleans()):
        kwargs['allow_equal'] = draw(st.booleans())
    if draw(st.booleans()):
        kwargs['let_end_none_if_start_none'] = draw(st.booleans())

    return kwargs


st_floats = partial(st.floats, allow_nan=False, allow_infinity=False)


@given(st.data())
def test_ranges_kwargs(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([None, st_floats]))
    kwargs = data.draw(ranges_kwargs(st_=st_))  # type: ignore

    min_start = kwargs.get('min_start')
    max_start = kwargs.get('max_start')
    assert sc(min_start) <= sc(max_start)

    min_end = kwargs.get('min_end')
    max_end = kwargs.get('max_end')
    assert sc(min_start) <= sc(min_end) <= sc(max_end)


@given(st.data())
@settings(max_examples=1000)
def test_ranges(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([None, st_floats]))
    kwargs = data.draw(ranges_kwargs(st_=st_))  # type: ignore

    args = (st_,) if st_ is not None else ()

    start, end = data.draw(ranges(*args, **kwargs))  # type: ignore

    allow_start_none = kwargs.get('allow_start_none', True)
    if not allow_start_none:
        assert start is not None

    let_end_none_if_start_none = kwargs.get('let_end_none_if_start_none', False)
    allow_end_none = kwargs.get('allow_end_none', True)
    if start is None and let_end_none_if_start_none:
        assert end is None
    elif not allow_end_none:
        assert end is not None

    allow_equal = kwargs.get('allow_equal', True)
    if allow_equal:
        assert sc(start) <= sc(end)
    else:
        assert sc(start) < sc(end)

    min_start = kwargs.get('min_start')
    max_start = kwargs.get('max_start')
    assert sc(min_start) <= sc(start) <= sc(max_start)

    min_end = kwargs.get('min_end')
    max_end = kwargs.get('max_end')
    assert sc(min_end) <= sc(end) <= sc(max_end)
