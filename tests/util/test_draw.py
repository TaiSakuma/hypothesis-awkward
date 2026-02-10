from typing import TypedDict, cast

from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from typing_extensions import Unpack

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util.draw import CountdownDrawer
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MIN_SIZE_EACH = 0
DEFAULT_MAX_SIZE_TOTAL = 10
DEFAULT_MAX_DRAWS = 100


def _sized_lists(*, min_size: int, max_size: int) -> st.SearchStrategy[list[int]]:
    '''A `_StWithMinMaxSize`-conforming callable for testing.'''
    return st.lists(st.integers(), min_size=min_size, max_size=max_size)


class CountdownDrawerKwargs(TypedDict, total=False):
    '''Options for `CountdownDrawer()`.'''

    min_size_each: int
    max_size_each: int
    max_size_total: int
    max_draws: int


@st.composite
def countdown_drawer_kwargs(draw: st.DrawFn) -> CountdownDrawerKwargs:
    '''Strategy for options for `CountdownDrawer()`.'''
    min_size_each, max_size_each = draw(st_ak.ranges(
        st.integers, min_start=0, max_start=10, max_end=50
    ))

    drawn = (
        ('min_size_each', min_size_each),
        ('max_size_each', max_size_each),
    )

    return draw(st.fixed_dictionaries(
        {k: st.just(v) for k, v in drawn if v is not None},
        optional={
            'max_size_total': st.integers(min_value=0, max_value=50),
            'max_draws': st.integers(min_value=0, max_value=200),
        },
    ).map(lambda d: cast(CountdownDrawerKwargs, d)))


@st.composite
def _exhaust(
    draw: st.DrawFn, **kwargs: Unpack[CountdownDrawerKwargs]
) -> tuple[int, list[list[int]]]:
    '''Draw from a `CountdownDrawer` until it returns `None`.'''
    draw_content = CountdownDrawer(draw, _sized_lists, **kwargs)

    total = 0
    results: list[list[int]] = []

    while True:
        result = draw_content()
        if result is None:
            break
        results.append(result)
        total += len(result)

    return total, results


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_countdown_drawer(data: st.DataObject) -> None:
    kwargs = data.draw(countdown_drawer_kwargs(), label='kwargs')
    total, results = data.draw(_exhaust(**kwargs), label='result')

    min_size_each = kwargs.get('min_size_each', DEFAULT_MIN_SIZE_EACH)
    max_size_each = kwargs.get('max_size_each')
    max_size_total = kwargs.get('max_size_total', DEFAULT_MAX_SIZE_TOTAL)
    max_draws = kwargs.get('max_draws', DEFAULT_MAX_DRAWS)

    assert total <= max_size_total
    assert len(results) <= max_draws

    for result in results:
        assert min_size_each <= len(result) <= sc(max_size_each)


def test_draw_max_size_total() -> None:
    '''Assert that the total can reach max_size_total.'''
    max_size_total = 10
    find(
        _exhaust(min_size_each=1, max_size_total=max_size_total),
        lambda r: r[0] == max_size_total,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_min_size_each() -> None:
    '''Assert that a draw can have exactly min_size_each elements.'''
    min_size_each = 5
    find(
        _exhaust(min_size_each=min_size_each, max_size_total=50),
        lambda r: any(len(result) == min_size_each for result in r[1]),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size_each() -> None:
    '''Assert that a draw can reach max_size_each elements.'''
    max_size_each = 5
    find(
        _exhaust(max_size_each=max_size_each, max_size_total=50),
        lambda r: any(len(result) == max_size_each for result in r[1]),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_draws() -> None:
    '''Assert that the number of draws can reach max_draws.'''
    max_draws = 200
    find(
        _exhaust(max_draws=max_draws),
        lambda r: len(r[1]) == max_draws,
        settings=settings(phases=[Phase.generate]),
    )
