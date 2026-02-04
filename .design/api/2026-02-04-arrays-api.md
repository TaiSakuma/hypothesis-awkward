# API Design: `arrays()` Strategy

**Date:** 2026-02-04
**Status:** Draft
**Author:** Claude (with developer collaboration)

## Overview

This document proposes an API for the `arrays()` strategy, which generates
`ak.Array` objects via direct Content constructors. The initial version supports
only `NumpyArray` as the layout node type. The design anticipates progressive
addition of `ListOffsetArray`, `RecordArray`, option types, unions, and more in
later iterations.

## Background

From the [UX research](./../research/2026-01-21-ux-interview-01.md),
[type system research](./../research/2026-01-21-type-system-research.md), and
[direct constructors research](./../research/2026-02-04-direct-constructors-research.md):

- The ultimate goal is an `arrays()` strategy that generates fully general
  Awkward Arrays with multiple options to control the layout, data types, missing
  values, masks, and other array attributes. `type` and `form` parameters are in
  scope but not mandatory
- Direct constructors produce real arrays in a single step (no roundtrip through
  forms or buffers)
- Constructor validation catches invalid nesting at construction time
- `ak.Array(layout)` wrapping is zero-cost
- Target users: Awkward developers, scikit-HEP tool developers, physicists

### Why Start with Direct Constructors

Three possible generation pipelines were considered:

1. **Type -> Form -> buffers -> array** (most abstract, most roundtrips)
2. **Form -> buffers -> array** (via `ak.from_buffers()`)
3. **Content layout -> array** (direct constructors, single step)

Pipeline 3 is the most direct: each Content constructor validates its inputs, so
the strategy can lean on built-in checks. It also exercises more internal code
paths than the canonical representations produced by `from_type()`.

### Why Start with NumpyArray Only

- `NumpyArray` is the leaf node -- every Awkward Array tree terminates in one
- The existing `numpy_arrays()` strategy already generates NumPy ndarrays;
  `arrays()` reuses this work
- Starting with a single node type validates the `arrays()` interface before
  adding recursive complexity
- Users get immediate value: `arrays()` with no arguments generates flat arrays,
  which is the most common case

## Design Goals

1. **Incremental extensibility**: Adding new node types must not break the
   existing interface
2. **Reuse existing strategies**: Build on `numpy_arrays()`, `supported_dtypes()`,
   etc.
3. **Familiar patterns**: Follow the parameter conventions established by
   `numpy_arrays()`, `types()`, and `numpy_forms()`
4. **Sensible defaults**: Works well out of the box; generates interesting arrays
   without configuration
5. **Composition with types/forms (future)**: The API should accommodate `type`
   and `form` parameters when those pipelines are connected

## Proposed API

### Main Strategy: `arrays()`

```python
@st.composite
def arrays(
    draw: st.DrawFn,

    # --- Leaf data control ---
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,

    # --- Size control ---
    max_length: int = 5,

    # --- Node type control (future expansion point) ---
    allow_list: bool = False,
    allow_record: bool = False,
    allow_option: bool = False,
    allow_union: bool = False,
    allow_string: bool = False,

    # --- Nesting control (future expansion point) ---
    max_depth: int = 3,
) -> ak.Array:
```

### Parameter Details

#### `dtypes`

Strategy for NumPy dtypes used in `NumpyArray` leaf nodes.

- If `None` (default), uses `supported_dtypes()`.
- If a strategy, draws from it.
- Follows the same convention as `numpy_types(dtypes=...)` and
  `numpy_forms(dtypes=...)`.

Note: Unlike the existing `numpy_arrays()` strategy, `arrays()` does not accept
a plain `np.dtype` value -- only a strategy or `None`. This is because
`arrays()` may draw multiple dtypes for different leaf nodes in a compound
structure (e.g., different record fields). Accepting a single dtype as a
convenience shorthand can be added later without breaking the interface.

#### `allow_nan`

Generate potentially `NaN`/`NaT` values for relevant dtypes.

- `False` (default): no `NaN`/`NaT` values.
- `True`: `NaN`/`NaT` values may appear.
- Same semantics as `numpy_arrays(allow_nan=...)`.

#### `max_length`

Maximum number of elements in the outermost array dimension (i.e., `len(result)`).

- Default: `5`.
- This controls the top-level length, not the total number of scalars.
- Name is `max_length` rather than `max_size` to distinguish it from
  `numpy_arrays(max_size=...)`, which limits the total number of elements across
  all dimensions.

Why `5` and not `10`? As more node types are added, compound arrays multiply the
amount of generated data. A smaller default avoids slow tests while still
providing meaningful coverage. This is the outermost dimension only; nested
structures can grow in depth.

#### `allow_list`, `allow_record`, `allow_option`, `allow_union`, `allow_string`

Control which Content node types are enabled.

- All default to `False` in the initial version.
- This means `arrays()` with no arguments generates flat `NumpyArray`-backed
  arrays -- the simplest case.
- As each node type is implemented, its default switches to `True`.
- Boolean flags follow the convention established in the `types()` API design.

When all flags are `False` (initial version), only `NumpyArray` is used: a 1-D
NumPy array wrapped in `ak.contents.NumpyArray` and then `ak.Array`.

#### `max_depth`

Maximum nesting depth for recursive node types.

- Default: `3`.
- `max_depth=0` forces leaf-only arrays (same as all `allow_*` flags being
  `False`).
- In the initial (NumpyArray-only) version, this parameter has no effect because
  there are no recursive node types.
- Included from the start so the interface is stable when list/record/option
  types are added.

### Return Type

`arrays()` returns `st.SearchStrategy[ak.Array]`.

**Decision:** Return `ak.Array`, not the raw `Content` layout.

**Rationale:**

- `ak.Array` is the user-facing type. Users write code against `ak.Array`, so
  the strategy should produce what they consume.
- `ak.Array(layout)` is zero-cost wrapping.
- Users needing the raw layout can access it via `result.layout`.
- Matches the precedent set by `from_numpy()`, which returns `ak.Array`.

### Parameters NOT Included

#### `allow_structured` / `allow_inner_shape`

Not included. In `numpy_arrays()`, `allow_structured=True` generates structured
NumPy arrays (with named fields), which `ak.from_numpy` converts to
`RecordArray`. In `arrays()`, records will be generated natively via
`RecordArray` when `allow_record=True`. Similarly, multi-dimensional NumpyArrays
(inner_shape) will be covered by `RegularArray` wrapping.

The `arrays()` strategy generates 1-D `NumpyArray` nodes only. Higher-dimensional
structure comes from nesting (lists, records), not from multi-dimensional NumPy
arrays.

#### `type` / `form`

Not included in the initial version. These will be added when the Type->Array
and Form->Array pipelines are connected:

```python
# Future signature extension:
def arrays(
    ...,
    type: ak.types.Type | st.SearchStrategy[ak.types.Type] | None = None,
    form: ak.forms.Form | st.SearchStrategy[ak.forms.Form] | None = None,
) -> st.SearchStrategy[ak.Array]:
```

When `type` or `form` is given, the strategy would generate arrays matching that
type or form. This can be added without breaking the current interface.

#### `allow_datetime`

Not included as a separate parameter. Users who want to exclude datetime dtypes
can filter via the `dtypes` parameter:

```python
non_datetime = st_ak.supported_dtypes().filter(lambda d: d.kind not in ('M', 'm'))
st_ak.constructors.arrays(dtypes=non_datetime)
```

This keeps the `arrays()` parameter list focused on structural concerns. The
`dtypes` parameter already provides this control.

## Implementation Strategy

### Initial Version (NumpyArray Only)

The initial implementation is straightforward:

```python
@st.composite
def arrays(draw, dtypes=None, allow_nan=False, max_length=5, **_future) -> ak.Array:
    if dtypes is None:
        dtypes = supported_dtypes()

    dtype = draw(dtypes)
    length = draw(st.integers(min_value=0, max_value=max_length))

    data = draw(
        st_np.arrays(
            dtype=dtype,
            shape=(length,),
            elements={'allow_nan': allow_nan},
        )
    )

    layout = ak.contents.NumpyArray(data)
    return ak.Array(layout)
```

Key differences from `numpy_arrays()`:

- Generates only 1-D arrays (no multi-dimensional shapes)
- Wraps in `NumpyArray` layout explicitly (not via `ak.from_numpy`)
- Uses `max_length` for the single dimension length
- Does not support structured dtypes (those become `RecordArray` later)

### Future Version Sketch (After Adding More Node Types)

```python
@st.composite
def arrays(draw, ..., max_depth=3) -> ak.Array:
    layout = draw(_contents(
        dtypes=dtypes,
        allow_nan=allow_nan,
        max_length=max_length,
        allow_list=allow_list,
        allow_record=allow_record,
        allow_option=allow_option,
        allow_union=allow_union,
        allow_string=allow_string,
        max_depth=max_depth,
    ))
    return ak.Array(layout)
```

Where `_contents()` is an internal recursive strategy that generates `Content`
layouts. The recursion pattern mirrors the `types()` implementation strategy:

```python
@st.composite
def _contents(draw, ..., max_depth, _forbidden=frozenset()):
    if max_depth <= 0:
        return draw(_numpy_arrays(...))

    strategies = [_numpy_arrays(...)]

    if allow_list and 'list' not in _forbidden:
        strategies.append(_list_offset_arrays(..., max_depth=max_depth - 1))

    if allow_option and 'option' not in _forbidden:
        strategies.append(_option_arrays(
            ..., max_depth=max_depth - 1,
            _forbidden=_forbidden | {'option', 'indexed', 'union'},
        ))

    # etc.

    return draw(st.one_of(*strategies))
```

The `_forbidden` set enforces the Content nesting constraints documented in the
direct constructors research (e.g., option nodes cannot wrap other option nodes).

## Supporting Strategies

### `numpy_array_contents()`

An internal strategy for generating `NumpyArray` Content nodes:

```python
@st.composite
def _numpy_array_contents(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    length: int | st.SearchStrategy[int] | None = None,
    max_length: int = 5,
) -> ak.contents.NumpyArray:
```

This is internal (not exported) because users interact with `arrays()`. The
`length` parameter is needed for recursive generation where parent nodes
determine child lengths (e.g., `RegularArray` needs content of length
`n * size`).

### Relationship to Existing Strategies

```text
Existing:
  supported_dtypes()  -->  numpy_arrays()   -->  from_numpy()  --> ak.Array
                       -->  numpy_types()                        (via ak.from_numpy)
                       -->  numpy_forms()

New:
  supported_dtypes()  -->  arrays()  --> ak.Array
                                        (via Content constructors)
```

`arrays()` does not wrap or delegate to `numpy_arrays()` or `from_numpy()`.
It generates data independently using `hypothesis.extra.numpy.arrays()` and
wraps it in `ak.contents.NumpyArray` directly. This avoids multi-dimensional
shapes and structured dtypes that `numpy_arrays()` produces, which are not
appropriate for the leaf nodes of a recursive content tree.

The existing `from_numpy()` and `numpy_arrays()` remain as-is. They serve a
different purpose: generating arrays specifically via the `ak.from_numpy()`
path, which includes structured arrays (RecordArray) and multi-dimensional
arrays. The `arrays()` strategy exercises the direct constructor path.

## Module Location

### Directory Structure

```text
src/hypothesis_awkward/strategies/
+-- constructors/
|   +-- __init__.py           # Re-exports arrays()
|   +-- arrays.py             # Main arrays() strategy
|   +-- numpy_.py             # _numpy_array_contents() internal helper
+-- builtins_/
+-- forms/
+-- misc/
+-- numpy/
+-- pandas/
+-- types/
+-- __init__.py               # Add arrays to public API
```

As more node types are added, the `constructors/` directory grows:

```text
src/hypothesis_awkward/strategies/constructors/
+-- __init__.py
+-- arrays.py                 # Main arrays() strategy
+-- numpy_.py                 # _numpy_array_contents()
+-- list_.py                  # _list_offset_array_contents(), etc. (future)
+-- record.py                 # _record_array_contents() (future)
+-- option.py                 # _option_array_contents() (future)
+-- union.py                  # _union_array_contents() (future)
+-- string.py                 # _string_array_contents() (future)
```

### Public API

The `constructors` subpackage is available as a namespace under `strategies/`:

```python
import hypothesis_awkward.strategies as st_ak

st_ak.constructors.arrays()
```

Not re-exported at the `st_ak` top level. This keeps `constructors` as a
distinct namespace, leaving room for alternative approaches (e.g.,
`st_ak.builders.arrays()`) without name collisions.

## Design Decisions

### 1. `allow_*` Flags Default to `False` Initially

**Decision:** All `allow_*` flags for unimplemented node types default to
`False`. As each node type is implemented, its default changes to `True`.

**Rationale:**

- Prevents users from passing `allow_list=True` and getting an error because
  lists are not yet implemented.
- The initial `arrays()` generates only flat arrays, which is explicit and
  predictable.
- Each new node type is a non-breaking additive change: the default becomes
  `True`, expanding what `arrays()` generates.

**Trade-off:** Users upgrading to a version where `allow_list` defaults to
`True` will start seeing list arrays where they previously saw only flat arrays.
This is acceptable because property-based testing should be robust to expanded
input space. Users who need stable behavior can pin `allow_list=False`.

### 2. `max_length` Instead of `max_size`

**Decision:** Use `max_length` to control the outermost dimension length.

**Rationale:**

- `max_size` in `numpy_arrays()` controls the total number of elements across
  all dimensions. This makes sense for multi-dimensional NumPy arrays.
- In `arrays()`, the outermost dimension length is the natural control point.
  Inner structure (list lengths, record field counts) will have separate
  controls.
- `max_length` is clearer: it is `len(result)`.
- Avoids confusion with `numpy_arrays(max_size=...)`.

### 3. Return `ak.Array`, Not `Content`

**Decision:** Return `ak.Array`.

**Rationale:**

- `ak.Array` is the user-facing type. Tests are written against `ak.Array`.
- `ak.Array(layout)` is zero-cost (no data copy).
- Raw layout is accessible via `result.layout`.
- Matches `from_numpy()` precedent.

### 4. No `allow_structured` Parameter

**Decision:** Omit `allow_structured`. Structured arrays (records) will come
from `allow_record=True`.

**Rationale:**

- In the `numpy_arrays()` -> `from_numpy()` path, structured NumPy dtypes
  produce RecordArray layouts. This was a side-effect of using `ak.from_numpy`.
- In `arrays()`, RecordArray is a first-class node type controlled by
  `allow_record`.
- Mixing structured dtypes with direct constructors would complicate the
  implementation without adding value.

### 5. `dtypes` Accepts Only Strategy or None, Not Plain `np.dtype`

**Decision:** `dtypes` accepts `st.SearchStrategy[np.dtype] | None`, not
`np.dtype | st.SearchStrategy[np.dtype] | None`.

**Rationale:**

- In compound arrays, multiple leaf nodes may need different dtypes.
- Accepting a plain `np.dtype` would fix all leaves to the same dtype, which is
  limiting.
- Users who want a single dtype can use `st.just(np.dtype('float64'))`.
- This can be relaxed later (accepting plain dtype as a convenience) without
  breaking the interface.

**Alternative considered:** Accepting plain `np.dtype` and wrapping in
`st.just()`, as `numpy_arrays(dtype=...)` does. Deferred to keep the initial
interface strict and revisit based on user feedback.

### 6. Separate `constructors/` Directory

**Decision:** Place `arrays()` in `strategies/constructors/`, not in
`strategies/` root or `strategies/numpy/`.

**Rationale:**

- The directory name reflects the approach (direct constructors), not the output.
  This leaves room for alternative approaches later (e.g., `builders/`,
  `from_types/`).
- `arrays()` is the main entry point, not specific to NumPy.
- It will grow to contain multiple internal modules (one per node type).
- Follows the pattern of `types/`, `forms/`, etc.
- `strategies/numpy/` is for the NumPy-specific path (`ak.from_numpy`).

### 7. Include `allow_*` Flags for Unimplemented Types from the Start

**Decision:** Include `allow_list`, `allow_record`, etc. in the initial
signature even though they are not yet implemented.

**Rationale:**

- Documents the intended scope of the strategy.
- Users can see what will be supported.
- The flags raise no errors (they just have no effect when `False`).
- Adding new parameters later would change the function signature, which could
  surprise users who rely on positional arguments (though keyword-only is
  preferred).

**Alternative considered:** Adding flags only when implemented. Rejected because
it makes the API surface unstable across versions.

## Usage Examples

### Basic Usage (Initial Version)

```python
import hypothesis_awkward.strategies as st_ak
from hypothesis import given

@given(a=st_ak.constructors.arrays())
def test_something(a):
    # a is a flat ak.Array backed by NumpyArray
    assert isinstance(a, ak.Array)
    assert isinstance(a.layout, ak.contents.NumpyArray)
```

### Specific Dtype

```python
import numpy as np
from hypothesis import strategies as st

@given(a=st_ak.constructors.arrays(dtypes=st.just(np.dtype('float64'))))
def test_float_arrays(a):
    assert a.layout.dtype == np.dtype('float64')
```

### Integer Dtypes Only

```python
int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')

@given(a=st_ak.constructors.arrays(dtypes=int_dtypes))
def test_integer_arrays(a):
    assert a.layout.dtype.kind == 'i'
```

### Allow NaN

```python
float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')

@given(a=st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True))
def test_nan_handling(a):
    # Test code that must handle NaN correctly
    ...
```

### Control Length

```python
@given(a=st_ak.constructors.arrays(max_length=100))
def test_larger_arrays(a):
    assert len(a) <= 100
```

### Future: With Lists and Records

```python
# Once allow_list and allow_record are implemented:
@given(a=st_ak.constructors.arrays(allow_list=True, allow_record=True, max_depth=2))
def test_nested_arrays(a):
    # a could be flat, a list of lists, a record, etc.
    assert isinstance(a, ak.Array)
```

### Future: With Type Constraint

```python
# Future: generate arrays matching a specific type
@given(a=st_ak.constructors.arrays(type=ak.types.ListType(ak.types.NumpyType('float64'))))
def test_typed_arrays(a):
    assert a.type.content == ak.types.NumpyType('float64')
```

## Testing Plan

Following the patterns in
[testing-patterns.md](./../../.claude/rules/testing-patterns.md):

### 1. TypedDict for kwargs

```python
class ArraysKwargs(TypedDict, total=False):
    '''Options for `arrays()` strategy.'''
    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_length: int
```

### 2. Strategy for kwargs with `st_ak.RecordDraws` and `st_ak.Opts`

```python
def arrays_kwargs() -> st.SearchStrategy[st_ak.Opts[ArraysKwargs]]:
    '''Strategy for options for `arrays()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_nan': st.booleans(),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
        .map(lambda d: cast(ArraysKwargs, d))
        .map(st_ak.Opts)
    )
```

### 3. Main property-based test

```python
@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` respects all its options.'''
    # Draw options
    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    a = data.draw(st_ak.constructors.arrays(**opts.kwargs), label='a')

    # Assert the result is always an ak.Array backed by NumpyArray
    assert isinstance(a, ak.Array)
    assert isinstance(a.layout, ak.contents.NumpyArray)

    # Assert the layout data is 1-D
    assert len(a.layout.data.shape) == 1

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    max_length = opts.kwargs.get('max_length', DEFAULT_MAX_LENGTH)

    note(f'{a=}')
    note(f'{a.layout.dtype=}')

    assert len(a) <= max_length

    match dtypes:
        case None:
            pass
        case st_ak.RecordDraws():
            drawn_dtype_names = {d.name for d in dtypes.drawn}
            assert a.layout.dtype.name in drawn_dtype_names

    if not allow_nan:
        assert not any_nan_nat_in_awkward_array(a)
```

### 4. Edge case reachability tests

```python
def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: len(a) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_length() -> None:
    '''Assert that arrays with max_length elements can be drawn.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: len(a) == DEFAULT_MAX_LENGTH,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_integer_dtype() -> None:
    '''Assert that integer dtype arrays can be drawn.'''
    int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')
    find(
        st_ak.constructors.arrays(dtypes=int_dtypes),
        lambda a: a.layout.dtype.kind == 'i',
        settings=settings(phases=[Phase.generate]),
    )
```

### 5. File structure for tests

```text
tests/strategies/constructors/
+-- __init__.py
+-- test_arrays.py
```

## Alternatives Considered

### Alternative A: Extend `from_numpy()` Instead

Expand `from_numpy()` to support more node types.

**Rejected because:**

- `from_numpy()` is specifically about the `ak.from_numpy()` path.
- Direct constructors exercise a different code path.
- Mixing concerns would make `from_numpy()` confusing.
- A separate `arrays()` strategy has a clearer purpose.

### Alternative B: Start from Forms

Generate `Form` objects first, then use `ak.from_buffers()` to produce arrays.

**Rejected because:**

- Requires buffer generation with complex invariants (sorted offsets, etc.).
- Two-step process (form + buffers) is harder to debug.
- Direct constructors validate at construction time.
- Can be added later as an alternative pipeline if needed.

### Alternative C: Start from Types

Generate `Type` objects, convert to `Form`, then to arrays.

**Rejected because:**

- Three-step pipeline (type -> form -> buffers -> array).
- Most indirection, hardest to debug.
- One type maps to multiple forms; need to resolve ambiguity.
- Can be added later via the `type` parameter.

### Alternative D: `max_size` Instead of `max_length`

Use `max_size` as in `numpy_arrays()`.

**Rejected because:**

- In nested structures, "total size" is ambiguous (total scalars? total
  elements at each level?).
- `max_length` is unambiguous: it is `len(result)`.
- Inner structure sizes will be controlled separately when those node types are
  added.

### Alternative E: Accept Plain `np.dtype` for `dtypes`

Accept `np.dtype | st.SearchStrategy[np.dtype] | None` as in `numpy_arrays()`.

**Deferred because:**

- In compound arrays, different leaves may need different dtypes.
- Accepting a plain dtype implies all leaves share it, which is limiting.
- Can be added later as a convenience without breaking changes.

## Open Questions

1. **Should `allow_empty` be a parameter?** Awkward Array has `EmptyArray` (a
   zero-length placeholder with `UnknownType`). Should `arrays()` be able to
   generate these? They are unusual and may cause issues in downstream code.
   Tentative answer: No, defer to a future `allow_empty` flag.

2. **Should `allow_indexed` be a parameter?** `IndexedArray` is type-transparent
   (it does not change the type of its content). It adds a level of indirection
   that exercises different code paths. This is valuable for testing Awkward
   internals but may be confusing for end users. Tentative answer: Yes, add it
   when implementing option types (since `IndexedOptionArray` is the main option
   representation and `IndexedArray` shares its machinery).

3. **When `type`/`form` parameters are added, how do they interact with
   `allow_*` flags?** Tentative answer: When `type` or `form` is given, it fully
   determines the structure and `allow_*` flags are ignored (similar to how
   `numpy_forms(type_=...)` ignores other parameters).

4. **Default `max_length` value?** `5` is proposed. The existing
   `numpy_arrays()` uses `max_size=10`. A smaller default is appropriate because
   compound arrays amplify size. However, for the initial flat-only version,
   `10` might be equally reasonable. The value should be a module-level constant
   (`DEFAULT_MAX_LENGTH = 5`) for easy adjustment.

## Next Steps

1. Implement `arrays()` with NumpyArray-only support
2. Add tests following the testing plan above
3. Export from `strategies/__init__.py`
4. Validate that the interface feels natural for the NumpyArray case
5. Design internal `_contents()` recursive strategy for adding
   `ListOffsetArray` in the next iteration
