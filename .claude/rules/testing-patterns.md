# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/numpy/test_numpy_arrays.py` (simple kwargs with `OptsChain`)
- `tests/strategies/forms/test_numpy_forms.py` (strategy-valued kwargs with
  `OptsChain` and `RecordDraws`)
- `tests/strategies/contents/test_content.py` (kwargs delegation via
  `OptsChain` chaining)

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

### Min/max pairs with `st_ak.ranges()`

Use `st_ak.ranges()` to generate `(min, max)` pairs where `min <= max` and
either may be `None`:

```python
min_size_each, max_size_each = draw(st_ak.ranges(
    st.integers, min_start=0, max_start=10, max_end=50
))
```

Include non-`None` values as required keys in `st.fixed_dictionaries`:

```python
drawn = (
    ('min_size_each', min_size_each),
    ('max_size_each', max_size_each),
)

return draw(st.fixed_dictionaries(
    {k: st.just(v) for k, v in drawn if v is not None},
    optional={...},
).map(lambda d: cast(MyKwargs, d)))
```

See `tests/util/test_draw.py` for a full example.

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

### Strategy-valued kwargs with `st_ak.OptsChain`

See `tests/strategies/forms/test_numpy_forms.py` for a full example.

When a parameter accepts both a concrete value and a strategy (e.g.,
`NumpyType | SearchStrategy[NumpyType] | None`), use `st_ak.OptsChain` to
register recorders via `chain.register()` so drawn values can be tracked in
assertions.

In the kwargs strategy, register recorders upfront, then use `st.just(recorder)`
in `st.fixed_dictionaries`:

```python
@st.composite
def numpy_forms_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[NumpyFormsKwargs]:
    if chain is None:
        chain = st_ak.OptsChain({})
    st_type = chain.register(st_ak.numpy_types())

    kwargs = draw(st.fixed_dictionaries(
        {
            'type_': st.one_of(
                st_ak.numpy_types(),        # concrete value
                st.just(st_type),           # strategy (tracked)
            ),
        },
    ))
    return chain.extend(cast(NumpyFormsKwargs, kwargs))
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

- `chain.register(strategy)` creates a `RecordDraws` wrapper and tracks it
- `st.just(recorder)` passes the recorder itself as the kwarg value
- `chain.extend(kwargs)` returns a new `OptsChain` with merged kwargs
- `reset()` clears all registered recorders (avoids stale state)
- `chain` parameter enables kwargs delegation between composable strategies
- `match` / `case` distinguishes concrete values from `st_ak.RecordDraws` in
  assertions

## 3. Main property-based test

Test that the strategy respects all its options:

```python
@settings(max_examples=200)
@given(data=st.data())
def test_numpy_arrays(data: st.DataObject) -> None:
    opts = data.draw(numpy_arrays_kwargs(), label='opts')
    opts.reset()
    result = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='result')
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

## 5. Optional bounds with `safe_compare`

When an option like `max_size` or `min_size` may be `None`, use
`safe_compare as sc` to write concise range assertions:

```python
from hypothesis_awkward.util.safe import safe_compare as sc

assert sc(min_size) <= len(result) <= sc(max_size)
```

`sc(None)` returns an object that is true for all inequality comparisons,
so `None` bounds are effectively ignored.

## 6. Global constants

Extract shared values like default parameters:

```python
DEFAULT_MAX_SIZE = 10
```
