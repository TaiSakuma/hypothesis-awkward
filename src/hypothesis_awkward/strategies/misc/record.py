from typing import Any, Generic, Mapping, Protocol, TypeVar

from hypothesis import strategies as st

T = TypeVar('T')
K = TypeVar('K', bound=Mapping[str, Any])


class _DrawableData(Protocol):
    '''Protocol for the ``data`` argument of ``do_draw``.'''

    def draw(self, strategy: st.SearchStrategy[T]) -> T: ...


class RecordDraws(st.SearchStrategy[T]):
    '''Wrap a strategy to store all drawn values.

    Examples
    --------
    >>> recorder = RecordDraws(st.integers())
    >>> value = recorder.example()
    >>> value in recorder.drawn
    True
    '''

    def __init__(self, base: st.SearchStrategy[T]) -> None:
        super().__init__()
        self.drawn: list[T] = []
        self._base = base

    def do_draw(self, data: _DrawableData) -> T:
        value = data.draw(self._base)
        self.drawn.append(value)
        return value


class OptsChain(Generic[K]):
    '''Drawn options with explicit recorder registration and kwargs merging.

    Owns recorder creation via ``register()`` and supports kwargs merging
    via ``extend()``.

    Examples
    --------
    >>> chain = OptsChain({'a': 1})
    >>> recorder = chain.register(st.integers())
    >>> child = chain.extend({'b': recorder})
    >>> child.kwargs == {'a': 1, 'b': recorder}
    True
    >>> child.reset()
    >>> recorder.drawn
    []
    '''

    def __init__(
        self,
        kwargs: K,
        _recorders: list[RecordDraws[Any]] | None = None,
    ) -> None:
        self._kwargs = kwargs
        self._recorders = _recorders if _recorders is not None else []

    @property
    def kwargs(self) -> K:
        return self._kwargs

    @property
    def recorders(self) -> list[RecordDraws[Any]]:
        return self._recorders

    def register(self, strategy: st.SearchStrategy[T]) -> RecordDraws[T]:
        '''Create a ``RecordDraws`` wrapper and track it for ``reset()``.'''
        recorder = RecordDraws(strategy)
        self._recorders.append(recorder)
        return recorder

    def extend(self, extra: Mapping[str, Any]) -> 'OptsChain[Any]':
        '''Return a new ``OptsChain`` with merged kwargs and a copy of recorders.'''
        return OptsChain(
            {**self._kwargs, **extra},
            _recorders=list(self._recorders),
        )

    def reset(self) -> None:
        '''Clear all recorded values from registered recorders.'''
        for r in self._recorders:
            r.drawn.clear()
