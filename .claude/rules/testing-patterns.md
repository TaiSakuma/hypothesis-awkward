# Strategy Testing Patterns

Reference implementation: `tests/strategies/numpy/test_numpy_arrays.py`

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

Use `find()` to verify that specific edge cases can be generated:

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
