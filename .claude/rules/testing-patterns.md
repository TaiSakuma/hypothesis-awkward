# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/contents/test_content.py` (base `OptsChain` pattern, all
  optional kwargs)
- `tests/strategies/numpy/test_numpy_arrays.py` (min/max pairs with `ranges()`)
- `tests/strategies/forms/test_numpy_forms.py` (mode selection for
  strategy-valued kwargs)
- `tests/strategies/misc/test_ranges.py` (dependent kwargs without `OptsChain`)

## 1. TypedDict for Strategy kwargs

Define a `TypedDict` that mirrors the strategy's parameters:

```python
class ContentsKwargs(TypedDict, total=False):
    '''Options for `contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    allow_record: bool
    allow_union: bool
    max_depth: int
```

## 2. Strategy for kwargs

### Base pattern with `OptsChain`

All kwargs strategies use `@st.composite` returning `st_ak.OptsChain[MyKwargs]`.
The `chain` parameter enables kwargs delegation between composable strategies.

```python
@st.composite
def contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ContentsKwargs]:
    '''Strategy for options for `contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_dtypes = chain.register(st_ak.supported_dtypes())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_dtypes),
                ),
                'max_size': st.integers(min_value=0, max_value=50),
                'allow_nan': st.booleans(),
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
                'allow_string': st.booleans(),
                'allow_bytestring': st.booleans(),
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
                'allow_record': st.booleans(),
                'allow_union': st.booleans(),
                'max_depth': st.integers(min_value=0, max_value=5),
            },
        )
    )

    return chain.extend(cast(ContentsKwargs, kwargs))
```

Key techniques:

- `@st.composite` with `chain` parameter for composability
- `chain.register(strategy)` creates a `RecordDraws` wrapper that tracks drawn
  values; pass it via `st.just(recorder)` as a strategy-valued kwarg
- `chain.extend(kwargs)` returns a new `OptsChain` with merged kwargs
- Use `st.fixed_dictionaries` with `optional` for independently drawn kwargs

### Min/max pairs with `st_ak.ranges()`

Use `st_ak.ranges()` to generate `(min, max)` pairs where `min <= max` and
either may be `None`:

```python
min_size, max_size = draw(
    st_ak.ranges(min_start=0, max_end=100, max_start=DEFAULT_MAX_SIZE)
)
```

Include non-`None` values as required keys in `st.fixed_dictionaries`:

```python
drawn = (
    ('min_size', min_size),
    ('max_size', max_size),
)

kwargs = draw(
    st.fixed_dictionaries(
        {k: st.just(v) for k, v in drawn if v is not None},
        optional={...},
    )
)

return chain.extend(cast(MyKwargs, kwargs))
```

See `tests/strategies/numpy/test_numpy_arrays.py` and
`tests/util/test_draw.py` for full examples.

### Dependent kwargs (without `OptsChain`)

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

### Mode selection for strategy-valued kwargs

See `tests/strategies/forms/test_numpy_forms.py` for a full example.

When a strategy has mutually exclusive parameter groups (e.g., `type_` mode vs
`dtypes` mode), define mode functions selected via `st.one_of`:

```python
@st.composite
def numpy_forms_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[NumpyFormsKwargs]:
    if chain is None:
        chain = st_ak.OptsChain({})
    st_type = chain.register(st_ak.numpy_types())
    st_dtypes = chain.register(st_ak.supported_dtypes())
    st_inner_shape = chain.register(_inner_shape_strategies())

    def type_mode() -> st.SearchStrategy[NumpyFormsKwargs]:
        return st.fixed_dictionaries(
            {
                'type_': st.one_of(
                    st_ak.numpy_types(),        # concrete value
                    st.just(st_type),           # strategy (tracked)
                ),
            },
        ).map(lambda d: cast(NumpyFormsKwargs, d))

    def dtypes_mode() -> st.SearchStrategy[NumpyFormsKwargs]:
        return st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(st.none(), st.just(st_dtypes)),
                'allow_datetime': st.booleans(),
                ...
            },
        ).map(lambda d: cast(NumpyFormsKwargs, d))

    kwargs = draw(st.one_of(type_mode(), dtypes_mode()))
    return chain.extend(kwargs)
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

- Register multiple recorders for different strategy-valued parameters
- `st.one_of(type_mode(), dtypes_mode())` selects between mutually exclusive
  parameter groups
- `reset()` clears all registered recorders before each draw (avoids stale
  state)
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
    n = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='n')
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
