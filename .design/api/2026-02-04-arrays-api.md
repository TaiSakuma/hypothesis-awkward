# API Design: `arrays()` Strategy

**Date:** 2026-02-04
**Status:** Implemented (initial version)
**Author:** Claude (with developer collaboration)

## Overview

This document describes the API for the `arrays()` strategy, which generates
`ak.Array` objects via direct Content constructors. The current implementation
generates arrays with `NumpyArray` leaf contents nested in `RegularArray`,
`ListOffsetArray`, and `ListArray` wrappers. The design anticipates progressive
addition of `RecordArray`, option types, unions, and more in later iterations.

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

### Why Start with Leaf Nodes (NumpyArray and EmptyArray)

- `NumpyArray` and `EmptyArray` are the leaf nodes -- every Awkward Array tree
  terminates in one
- The existing `numpy_arrays()` strategy already generates NumPy ndarrays;
  `arrays()` reuses this work via `numpy_array_contents()`
- `EmptyArray` is a zero-length placeholder with `UnknownType` -- useful for
  testing edge cases with empty/unknown-typed arrays
- Starting with leaf node types validates the `arrays()` interface before
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

## API

> **Note:** Signatures below omit `*` separators. See
> [positional-keyword-convention](../notes/2026-02-12-positional-keyword-convention.md)
> for the keyword-only convention adopted after this document was written.
> `arrays()` falls in Group B (all keyword-only): `(draw, *, dtypes=None, ...)`.

### Main Strategy: `arrays()`

```python
@st.composite
def arrays(
    draw: st.DrawFn,
    *,
    # --- Leaf data control ---
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,

    # --- Size control ---
    max_size: int = 10,

    # --- Leaf type control ---
    allow_numpy: bool = True,
    allow_empty: bool = True,

    # --- Nesting type control ---
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,

    # --- Nesting control ---
    max_depth: int = 5,
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

#### `max_size`

Maximum total number of scalar values in the generated array.

- Default: `10`.
- Controls the total scalar budget across all leaf nodes, not just the outermost
  dimension length. `EmptyArray` leaves consume no budget (length 0).
- Uses a "budgeted leaf" approach: each leaf draws up to the remaining budget,
  preventing compound arrays from growing unboundedly.

#### `allow_numpy`, `allow_empty`

Control which leaf Content node types are enabled.

- Both default to `True`.
- `allow_numpy`: Generate `NumpyArray` leaves (primitive data).
- `allow_empty`: Generate `EmptyArray` leaves (zero-length, `UnknownType`).
  `EmptyArray` is unaffected by `dtypes` and `allow_nan`.
- At least one leaf type must be enabled; disabling both raises `ValueError`.

#### `allow_regular`, `allow_list_offset`, `allow_list`

Control which structural Content node types are enabled.

- All default to `True`.
- `allow_regular`: Generate `RegularArray` wrappers (fixed-size lists).
- `allow_list_offset`: Generate `ListOffsetArray` wrappers (variable-length lists
  via offsets).
- `allow_list`: Generate `ListArray` wrappers (variable-length lists via
  starts/stops).
- When all three are `False`, only flat leaf arrays are generated (`NumpyArray`
  and/or `EmptyArray`).

Future node type flags (not yet implemented): `allow_record`, `allow_option`,
`allow_union`, `allow_string`.

#### `max_depth`

Maximum nesting depth for structural wrappers.

- Default: `5`.
- `max_depth=0` forces leaf-only arrays (flat `NumpyArray`).
- The strategy draws a random depth between 0 and `max_depth`, then applies that
  many randomly chosen wrappers (`RegularArray`, `ListOffsetArray`, `ListArray`).

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

## Implementation

### Two-layer Architecture

The implementation is split across two packages:

- **`contents/content.py`**: `contents()` strategy generates `ak.contents.Content`
  layouts using a wrappers pattern
- **`constructors/array_.py`**: `arrays()` is a thin wrapper that calls
  `contents()` and wraps the result in `ak.Array`

### `contents()` Strategy

The `contents()` strategy in `contents/content.py` uses a wrappers pattern:

1. **Leaf strategy**: `leaf_contents()` generates a `NumpyArray` or `EmptyArray`
   Content with a scalar budget managed by `CountdownDrawer`
2. A random depth (0 to `max_depth`) is drawn
3. Nesting functions (`regular_array_contents`, `list_offset_array_contents`,
   `list_array_contents`) are chosen randomly for each depth level
4. Nesting functions are applied from innermost to outermost

### `arrays()` Strategy

`arrays()` in `constructors/array_.py` simply forwards all parameters to
`contents()` and wraps the result:

```python
@st.composite
def arrays(draw, ...) -> ak.Array:
    layout = draw(st_ak.contents.contents(...))
    return ak.Array(layout)
```

### Future Expansion

When adding `RecordArray`, option types, and unions, the wrappers pattern will
need to evolve to handle content nesting constraints (e.g., option nodes cannot
wrap other option nodes). This may require adding constraint tracking to the
wrappers pattern.

### Relationship to Existing Strategies

```text
Existing:
  supported_dtypes()  -->  numpy_arrays()   -->  from_numpy()  --> ak.Array
                       -->  numpy_types()                        (via ak.from_numpy)
                       -->  numpy_forms()

New:
  supported_dtypes()  -->  numpy_array_contents()  -->  contents()  --> Content
                                                        arrays()    --> ak.Array
```

## Module Location

### Directory Structure

```text
src/hypothesis_awkward/strategies/
+-- contents/
|   +-- __init__.py           # Re-exports contents() and individual content strategies
|   +-- content.py            # contents() — top-level content layout strategy
|   +-- leaf.py               # leaf_contents() — leaf node strategy (NumpyArray | EmptyArray)
|   +-- numpy_array.py        # numpy_array_contents()
|   +-- empty_array.py        # empty_array_contents()
|   +-- regular_array.py      # regular_array_contents()
|   +-- list_offset_array.py  # list_offset_array_contents()
|   +-- list_array.py         # list_array_contents()
+-- constructors/
|   +-- __init__.py           # Re-exports arrays()
|   +-- array_.py             # arrays() — delegates to contents.contents()
+-- builtins_/
+-- forms/
+-- misc/
+-- numpy/
+-- pandas/
+-- types/
+-- __init__.py               # Re-exports arrays via constructors namespace
```

`arrays()` in `constructors/array_.py` is a thin wrapper that forwards all
arguments to `contents.contents()` and wraps the result in `ak.Array`. The
`contents/` package contains the layout generation logic. As more node types are
added, new content strategies will be added to `contents/`:

```text
src/hypothesis_awkward/strategies/contents/
+-- __init__.py
+-- content.py                # Main contents() strategy (recursive composition)
+-- leaf.py                   # leaf_contents() — leaf node strategy
+-- numpy_array.py            # numpy_array_contents()
+-- empty_array.py            # empty_array_contents()
+-- regular_array.py          # regular_array_contents()
+-- list_offset_array.py      # list_offset_array_contents()
+-- list_array.py             # list_array_contents()
+-- record_array.py           # record_array_contents() (future)
+-- option.py                 # option content strategies (future)
+-- union_array.py            # union_array_contents() (future)
+-- string.py                 # string content strategies (future)
```

### Public API

The `constructors` subpackage is available as a namespace under `strategies/`:

```python
import hypothesis_awkward.strategies as st_ak

st_ak.constructors.arrays()
```

Not re-exported at the `st_ak` top level. The `constructors` module is imported
as a namespace (`from . import constructors`), so it is accessed as
`st_ak.constructors.arrays()`.

## Design Decisions

### 1. Implemented `allow_*` Flags Default to `True`

**Decision:** `allow_regular`, `allow_list_offset`, and `allow_list` default to
`True`. Future flags for unimplemented node types will be added when those node
types are implemented.

**Rationale:**

- Maximizes coverage by default -- `arrays()` with no arguments generates a
  variety of structural layouts.
- Users who want flat arrays can set all flags to `False`.
- New `allow_*` flags will be added (defaulting to `True`) as new node types are
  implemented.

### 2. `max_size` (Total Scalar Budget)

**Decision:** Use `max_size` to control the total number of scalar values across
all leaf `NumpyArray` nodes.

**Rationale:**

- In nested structures, controlling the outermost dimension length alone does
  not bound total array size.
- A scalar budget provides a natural cap that works regardless of nesting depth.
- The "budgeted leaf" approach (closure with `remaining` counter) ensures the
  budget is respected across multiple leaf draws.

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

### 7. Add `allow_*` Flags Only When Implemented

**Decision:** Only include `allow_*` flags for node types that are actually
implemented. Currently: `allow_numpy`, `allow_empty` (leaf types),
`allow_regular`, `allow_list_offset`, `allow_list` (nesting types).

**Rationale:**

- Avoids dead parameters that have no effect.
- The signature grows as features are added, which is natural for an
  experimental strategy.
- Keyword-only arguments make this a non-breaking change.

## Usage Examples

### Basic Usage

```python
import hypothesis_awkward.strategies as st_ak
from hypothesis import given

@given(a=st_ak.constructors.arrays())
def test_something(a):
    # a is an ak.Array, possibly nested with RegularArray/ListOffsetArray/ListArray
    assert isinstance(a, ak.Array)
```

### Specific Dtype

```python
import numpy as np
from hypothesis import strategies as st
from hypothesis_awkward.util import iter_numpy_arrays

@given(a=st_ak.constructors.arrays(dtypes=st.just(np.dtype('float64'))))
def test_float_arrays(a):
    for leaf in iter_numpy_arrays(a):
        assert leaf.dtype == np.dtype('float64')
```

### Integer Dtypes Only

```python
int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')

@given(a=st_ak.constructors.arrays(dtypes=int_dtypes))
def test_integer_arrays(a):
    for leaf in iter_numpy_arrays(a):
        assert leaf.dtype.kind == 'i'
```

### Allow NaN

```python
float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')

@given(a=st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True))
def test_nan_handling(a):
    # Test code that must handle NaN correctly
    ...
```

### Control Size

```python
@given(a=st_ak.constructors.arrays(max_size=100))
def test_larger_arrays(a):
    # Total scalar count across all leaves is at most 100
    ...
```

### Flat Arrays Only

```python
@given(a=st_ak.constructors.arrays(
    allow_regular=False, allow_list_offset=False, allow_list=False,
))
def test_flat_arrays(a):
    assert isinstance(a.layout, (ak.contents.NumpyArray, ak.contents.EmptyArray))
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

### `contents()` tests (`tests/strategies/contents/test_content.py`)

The `contents()` strategy has comprehensive property-based and reachability tests
that verify all options are respected: `max_size` (total scalars), per-type
gating (`allow_regular`, `allow_list_offset`, `allow_list`), dtypes via leaf
arrays, `allow_nan`, and `max_depth`. Edge case reachability tests use `find()`
to verify the strategy can produce empty content, NaN values, specific dtypes,
maximum depth, nested structures, and edge cases for each list type.

### `arrays()` tests (`tests/strategies/constructors/test_array.py`)

Since `arrays()` is a thin wrapper around `contents()`, the test mocks
`contents()` and verifies:

1. All kwargs are forwarded to `contents()` (via `assert_called_once_with`)
2. The result is an `ak.Array` wrapping the content returned by `contents()`
   (via identity check on `.layout`)

### File structure for tests

```text
tests/strategies/contents/
+-- __init__.py
+-- test_content.py           # contents() strategy tests
+-- test_empty_array.py       # empty_array_contents() tests
+-- test_numpy_array.py       # numpy_array_contents() tests
+-- test_regular_array.py     # regular_array_contents() tests
+-- test_list_offset_array.py # list_offset_array_contents() tests
+-- test_list_array.py        # list_array_contents() tests

tests/strategies/constructors/
+-- __init__.py
+-- test_array.py             # arrays() strategy tests (mocks contents())
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

### Alternative D: `max_length` Instead of `max_size`

Control the outermost dimension length rather than total scalars.

**Rejected because:**

- In nested structures, outermost length alone does not bound total array size.
- A scalar budget (`max_size`) provides a natural cap regardless of nesting depth.
- The "budgeted leaf" approach made `max_size` straightforward to implement.

### Alternative E: Accept Plain `np.dtype` for `dtypes`

Accept `np.dtype | st.SearchStrategy[np.dtype] | None` as in `numpy_arrays()`.

**Deferred because:**

- In compound arrays, different leaves may need different dtypes.
- Accepting a plain dtype implies all leaves share it, which is limiting.
- Can be added later as a convenience without breaking changes.

## Open Questions

1. ~~**Should `allow_empty` be a parameter?**~~ **Resolved.** Yes, `allow_empty`
   is now implemented (default `True`). `EmptyArray` leaves are generated via
   `leaf_contents()` when `allow_empty=True` and `min_size == 0`.

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

## Completed

1. ~~Implement `arrays()` with NumpyArray-only support~~ ✓
2. ~~Add `RegularArray`, `ListOffsetArray`, `ListArray` support~~ ✓
3. ~~Add scalar budget approach (`max_size`)~~ ✓
4. ~~Add `max_depth` parameter~~ ✓
5. ~~Add tests following the testing plan~~ ✓
6. ~~Export from `strategies/__init__.py`~~ ✓

## Next Steps

1. Add `RecordArray` support (`allow_record`)
2. Add option type support (`allow_option`) -- `IndexedOptionArray`,
   `ByteMaskedArray`, `BitMaskedArray`, `UnmaskedArray`
3. Add `UnionArray` support (`allow_union`)
4. Add string/bytestring support (`allow_string`, `allow_bytestring`) — see
   [string-bytestring-api](./../api/2026-02-13-string-bytestring-api.md) for
   the proposed design
5. Implement content nesting constraint enforcement
6. Consider connecting `type`/`form` parameters
