# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/numpy/test_numpy_arrays.py` (simple kwargs)
- `tests/strategies/forms/test_numpy_forms.py` (strategy-valued kwargs with
  `RecordDraws`)

## 1. TypedDict for Strategy kwargs

Define a `TypedDict` that mirrors the strategy's parameters:

```python
class NumpyArraysKwargs(TypedDict, total=False):
    '''Options for `numpy_arrays()` strategy.'''
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    max_size: int
```

## 2. Strategy for kwargs

### Simple case: all optional, independent kwargs

Use `st.fixed_dictionaries` with `optional`:

```python
def numpy_arrays_kwargs() -> st.SearchStrategy[NumpyArraysKwargs]:
    '''Strategy for options for `numpy_arrays()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(st.none(), st_ak.supported_dtypes()),
            'allow_structured': st.booleans(),
            'allow_nan': st.booleans(),
            'max_size': st.integers(min_value=0, max_value=100),
        },
    ).map(lambda d: cast(NumpyArraysKwargs, d))
```

### Complex case: dependent kwargs

See `tests/strategies/misc/test_ranges.py` for a full example.

When kwargs depend on each other, use `@st.composite` and `flatmap`:

```python
@st.composite
def ranges_kwargs(
    draw: st.DrawFn, st_: StMinMaxValuesFactory[T] | None = None
) -> RangesKwargs[T]:
    if st_ is None:
        st_ = st.integers

    # Generate dependent values using flatmap
    min_start, max_start = draw(min_max_starts(st_=st_))
    min_end, max_end = draw(min_max_ends(st_=st_, min_start=min_start))

    # Collect non-None values as required kwargs
    drawn = (
        ('min_start', min_start),
        ('max_start', max_start),
        ('min_end', min_end),
        ('max_end', max_end),
    )

    # Mix required (non-None drawn values) and optional kwargs
    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'allow_start_none': st.booleans(),
                'allow_end_none': st.booleans(),
                'let_end_none_if_start_none': st.booleans(),
                'allow_equal': st.booleans(),
            },
        )
    )

    return cast(RangesKwargs[T], kwargs)
```

Key techniques:

- `@st.composite` allows multiple `draw()` calls with dependencies
- `flatmap` chains dependent strategies (e.g., max depends on min)
- Mix required and optional in `st.fixed_dictionaries`

### Strategy-valued kwargs with `st_ak.RecordDraws`

See `tests/strategies/forms/test_numpy_forms.py` for a full example.

When a parameter accepts both a concrete value and a strategy (e.g.,
`NumpyType | SearchStrategy[NumpyType] | None`), use `st_ak.RecordDraws` to
wrap strategies so drawn values can be tracked in assertions.

In the kwargs strategy, use `st.just(st_ak.RecordDraws(...))` to pass the
recorder as a value:

```python
'type_': st.one_of(
    st_ak.numpy_types(),                                 # concrete value
    st.just(st_ak.RecordDraws(st_ak.numpy_types())),    # strategy (tracked)
),
```

Wrap kwargs in `st_ak.Opts[K]` with `reset()` to clear recorders between draws:

```python
def numpy_forms_kwargs() -> st.SearchStrategy[st_ak.Opts[NumpyFormsKwargs]]:
    return (
        st.one_of(type_mode(), dtypes_mode())
        .map(st_ak.Opts)
    )
```

In the test, call `reset()` before drawing and use `match` for assertions:

```python
opts = data.draw(numpy_forms_kwargs(), label='opts')
opts.reset()
result = data.draw(st_ak.numpy_forms(**opts.kwargs), label='result')

match type_:
    case ak.types.NumpyType():
        assert result.primitive == type_.primitive
    case st_ak.RecordDraws():
        drawn_primitives = {t.primitive for t in type_.drawn}
        assert result.primitive in drawn_primitives
```

Key techniques:

- `st_ak.RecordDraws` records values drawn from a wrapped strategy
- `st.just(st_ak.RecordDraws(...))` passes the recorder itself as the kwarg
  value
- `st_ak.Opts[K]` is a generic wrapper; `reset()` clears recorded values before
  each draw (avoids stale state)
- `match` / `case` distinguishes concrete values from `st_ak.RecordDraws` in
  assertions

## 3. Main property-based test

Test that the strategy respects all its options:

```python
@settings(max_examples=200)
@given(data=st.data())
def test_numpy_arrays(data: st.DataObject) -> None:
    kwargs = data.draw(numpy_arrays_kwargs(), label='kwargs')
    result = data.draw(st_ak.numpy_arrays(**kwargs), label='result')
    # Assert options were effective...
```

## 4. Edge case reachability tests using `find()`

The main property test asserts that invariants hold for every draw — it tests
universal properties. It cannot assert that something is ever produced. `find()`
tests the opposite: that there exists a draw satisfying a predicate.

```python
def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(),
        lambda a: math.prod(a.shape) == 0,
        settings=settings(phases=[Phase.generate]),
    )
```

- Use `phases=[Phase.generate]` to skip shrinking (faster)
- Use `max_examples=2000` for rare conditions
- Use specific dtypes to target relevant types (e.g., `st_np.floating_dtypes()`
  for NaN tests)

## 5. Global constants

Extract shared values like default parameters:

```python
DEFAULT_MAX_SIZE = 10
```
