from collections.abc import Callable, Sized
from typing import Protocol, TypeVar

from hypothesis import strategies as st

from hypothesis_awkward.util.safe import safe_min

_C_co = TypeVar('_C_co', covariant=True)
_T = TypeVar('_T', bound=Sized)


class _StWithMinMaxSize(Protocol[_C_co]):
    '''A callable that takes ``min_size`` and ``max_size`` keywords.'''

    def __call__(self, *, min_size: int, max_size: int) -> st.SearchStrategy[_C_co]: ...


def CountdownDrawer(
    draw: st.DrawFn,
    st_: _StWithMinMaxSize[_T],
    min_size_each: int = 0,
    max_size_each: int | None = None,
    max_size_total: int = 10,
    max_draws: int = 100,
) -> Callable[[], _T | None]:
    '''Create a draw function that counts down from ``max_size_total``.

    Each call draws from ``st_`` and subtracts the length of the result
    from the remaining count. Returns ``None`` once the budget is exhausted,
    too small to satisfy ``min_size_each``, or the draw limit is reached.

    Parameters
    ----------
    draw
        The Hypothesis draw function.
    st_
        A callable that accepts ``min_size`` and ``max_size`` keyword
        arguments and returns a strategy.
    min_size_each
        Minimum number of elements in each draw.
    max_size_each
        Maximum number of elements in each draw. If ``None``, only
        ``max_size_total`` limits the size.
    max_size_total
        Total element budget shared across all draws.
    max_draws
        Maximum number of non-None draws.
    '''

    max_size_total = draw(st.integers(min_value=0, max_value=max_size_total))

    def _draw_content() -> _T | None:
        nonlocal max_size_total, max_draws
        if max_draws == 0:
            return None
        if max_size_total == 0 or max_size_total < min_size_each:
            return None
        max_size = safe_min((max_size_each, max_size_total))
        assert max_size is not None
        result = draw(st_(min_size=min_size_each, max_size=max_size))
        max_size_total -= len(result)
        max_draws -= 1
        return result

    return _draw_content
