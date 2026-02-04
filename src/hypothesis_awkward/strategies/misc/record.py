from typing import Any, TypeVar

from hypothesis import strategies as st

T = TypeVar('T')


class RecordDraws(st.SearchStrategy[T]):
    '''Wrap a strategy to store all drawn values.'''

    def __init__(self, base: st.SearchStrategy[T]) -> None:
        super().__init__()
        self.drawn: list[T] = []
        self._base = base

    def do_draw(self, data: Any) -> T:
        value = data.draw(self._base)
        self.drawn.append(value)
        return value
