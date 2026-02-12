# API Design: `contents/` Strategies

**Date:** 2026-02-12
**Status:** Implemented (initial version)
**Author:** Claude (with developer collaboration)

## Overview

This document describes the API for the seven strategies in the `contents/`
package, which generate `ak.contents.Content` layout objects for property-based
testing. The `contents/` package is the layout generation layer: it produces
Awkward Array internal structures (`NumpyArray`, `EmptyArray`, `RegularArray`,
`ListOffsetArray`, `ListArray`) that the `constructors/` package wraps in
`ak.Array`. See [arrays-api.md](./2026-02-04-arrays-api.md) for the
`arrays()` strategy that consumes `contents()`.

## Background

From the [arrays API design](./2026-02-04-arrays-api.md) and
[direct constructors research](./../research/2026-02-04-direct-constructors-research.md):

- The generation pipeline is: Content layout -> `ak.Array` (single step)
- Direct constructors validate inputs at construction time
- `arrays()` is a thin wrapper around `contents()`, forwarding all parameters
  and calling `ak.Array(layout)`
- The `contents/` package encapsulates the layout generation logic, keeping
  `constructors/` focused on the public API
- Target users: Awkward developers, scikit-HEP tool developers, physicists

## Design Goals

1. **Two-layer separation**: Layout generation (`contents/`) is distinct from
   the public `ak.Array` API (`constructors/`)
2. **Incremental extensibility**: New node types (records, options, unions) are
   added as new modules without breaking existing strategies
3. **Composability**: Wrapper strategies accept `Content | Strategy | None`,
   enabling both standalone use and composition via `contents()`
4. **Scalar budgeting**: Total scalar count is bounded across all leaves,
   preventing compound arrays from growing unboundedly
5. **Sensible defaults**: All strategies work well with no arguments

## API

> **Note:** Signatures below show the current code. See
> [positional-keyword-convention](../notes/2026-02-12-positional-keyword-convention.md)
> for the keyword-only convention adopted after this document was written.
> `contents()`, `numpy_array_contents()`, and `leaf_contents()` fall in Group B
> (all keyword-only). The wrapper strategies (`regular_array_contents`,
> `list_offset_array_contents`, `list_array_contents`) fall in Group A:
> `content` is positional, future config after `*`.

### `contents()`

Top-level recursive strategy that composes leaf and wrapper strategies into
nested content layouts.

```python
@st.composite
def contents(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> Content:
```

#### Parameters

- **`dtypes`** — Strategy for NumPy scalar dtypes used in `NumpyArray` leaves.
  If `None` (default), uses `supported_dtypes()`. Follows the same convention as
  `arrays(dtypes=...)`. Note: only a strategy or `None`, not a plain `np.dtype`
  (see [arrays-api.md](./2026-02-04-arrays-api.md), Design Decision 5).

- **`max_size`** — Maximum total number of scalar values across all leaf
  `NumpyArray` nodes. Default: `10`. Controls the total scalar budget, not just
  the outermost dimension length. `EmptyArray` leaves consume no budget (length
  0). Internally managed by `CountdownDrawer`.

- **`allow_nan`** — Generate potentially `NaN`/`NaT` values for relevant
  dtypes. Default: `False`.

- **`allow_numpy`** — Generate `NumpyArray` leaves. Default: `True`.

- **`allow_empty`** — Generate `EmptyArray` leaves. Default: `True`.
  `EmptyArray` is unaffected by `dtypes` and `allow_nan`. At least one of
  `allow_numpy` or `allow_empty` must be `True`; disabling both raises
  `ValueError`.

- **`allow_regular`** — Generate `RegularArray` wrappers. Default: `True`.

- **`allow_list_offset`** — Generate `ListOffsetArray` wrappers. Default:
  `True`.

- **`allow_list`** — Generate `ListArray` wrappers. Default: `True`. When all
  three wrapper flags are `False`, only flat leaf arrays are generated.

- **`max_depth`** — Maximum nesting depth for structural wrappers. Default: `5`.
  `max_depth=0` forces leaf-only arrays. The strategy draws a random depth
  between 0 and `max_depth`, then randomly chooses a wrapper for each level.

#### No `min_size`

`contents()` does not expose `min_size`. Per-leaf `min_size` is managed
internally by `CountdownDrawer`, which distributes the scalar budget across
draws and raises the effective floor to satisfy total minimums.

### `leaf_contents()`

Leaf selector that returns either a `NumpyArray` or `EmptyArray`. All parameters
are keyword-only because there is no "what" parameter — everything is
configuration.

```python
def leaf_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
    allow_numpy: bool = True,
    allow_empty: bool = True,
) -> st.SearchStrategy[NumpyArray | EmptyArray]:
```

#### Parameters

- **`dtypes`** — Same as `contents(dtypes=...)`. Forwarded to
  `numpy_array_contents()`.

- **`allow_nan`** — Same as `contents(allow_nan=...)`. Forwarded to
  `numpy_array_contents()`.

- **`min_size`** — Minimum number of scalar values. Default: `0`. When
  `min_size > 0`, `EmptyArray` is excluded (since `EmptyArray` always has length
  0).

- **`max_size`** — Maximum number of scalar values. Default: `10`. Forwarded to
  `numpy_array_contents()`.

- **`allow_numpy`** — Generate `NumpyArray`. Default: `True`. At least one of
  `allow_numpy` or `allow_empty` must be `True`; disabling both raises
  `ValueError`.

- **`allow_empty`** — Generate `EmptyArray`. Default: `True`. Only effective
  when `min_size == 0`.

#### Behavior

Uses `st.one_of()` to select between enabled leaf types:

- `allow_numpy=True` → includes `numpy_array_contents(dtypes, allow_nan,
  min_size=min_size, max_size=max_size)`
- `allow_empty=True` and `min_size == 0` → includes `empty_array_contents()`

### `numpy_array_contents()`

Generates 1-D `NumpyArray` content by drawing a NumPy array via
`numpy_arrays()` and wrapping it in `ak.contents.NumpyArray`.

```python
def numpy_array_contents(
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
```

#### Parameters

- **`dtypes`** — Strategy for NumPy scalar dtypes. If `None`, uses
  `supported_dtypes()`. Forwarded as `dtype` to `numpy_arrays()`.

- **`allow_nan`** — Same as `numpy_arrays(allow_nan=...)`.

- **`min_size`** — Minimum number of elements. Default: `0`.

- **`max_size`** — Maximum number of elements. Default: `10`.

#### Implementation

Delegates to `numpy_arrays(dtype=dtypes, allow_structured=False, allow_nan=...,
max_dims=1, min_size=..., max_size=...).map(ak.contents.NumpyArray)`. Always
generates 1-D arrays with no structured dtypes — higher-dimensional structure
comes from nesting wrappers.

### `empty_array_contents()`

Returns a constant strategy that always produces `EmptyArray()`.

```python
def empty_array_contents() -> st.SearchStrategy[ak.contents.EmptyArray]:
```

No parameters. Returns `st.just(ak.contents.EmptyArray())`. `EmptyArray` has
Awkward type `unknown` and length 0.

### `regular_array_contents()`

Wraps child content in `RegularArray`.

```python
MAX_REGULAR_SIZE = 5

@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> Content:
```

#### Parameters

- **`content`** — The child content to wrap. Accepts three forms:
  - `None` (default): draws from `contents()` (recursive)
  - `st.SearchStrategy[Content]`: draws from the strategy
  - `Content`: uses directly

#### Behavior

The `size` parameter of `RegularArray` is chosen to evenly divide the content
length. When content length is 0, `size` is drawn from
`[0, MAX_REGULAR_SIZE]`; if `size` is also 0, `zeros_length` is drawn from
`[0, MAX_REGULAR_SIZE]`.

#### Constants

- `MAX_REGULAR_SIZE = 5` — upper bound for the `size` divisor and for
  `zeros_length`

### `list_offset_array_contents()`

Wraps child content in `ListOffsetArray`.

```python
MAX_LIST_LENGTH = 5

@st.composite
def list_offset_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> Content:
```

#### Parameters

- **`content`** — Same three-form dispatch as `regular_array_contents()`.

#### Behavior

Draws `n` (number of lists) from `[0, MAX_LIST_LENGTH]`, then generates sorted
split points to partition the content into `n` sublists. The resulting offsets
array is monotonically non-decreasing and covers all content elements.

#### Constants

- `MAX_LIST_LENGTH = 5` — upper bound for the number of sublists

### `list_array_contents()`

Wraps child content in `ListArray`. Identical logic to
`list_offset_array_contents()` but produces separate `starts` and `stops` index
arrays instead of a single `offsets` array.

```python
MAX_LIST_LENGTH = 5

@st.composite
def list_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> Content:
```

#### Parameters

- **`content`** — Same three-form dispatch as `regular_array_contents()`.

#### Behavior

Same partitioning logic as `list_offset_array_contents()`. The offsets array is
split into `starts = offsets[:-1]` and `stops = offsets[1:]`.

#### Constants

- `MAX_LIST_LENGTH = 5` — upper bound for the number of sublists

## Design Decisions

### 1. Two-Layer Architecture

**Decision:** Separate `contents/` (layout generation) from `constructors/`
(`ak.Array` wrapping).

**Rationale:**

- `contents()` produces `Content` layouts, which are the internal tree nodes of
  Awkward Arrays. `arrays()` in `constructors/` is a thin wrapper that calls
  `contents()` and wraps the result in `ak.Array`
- This separation keeps the layout logic testable independently of the public
  API
- Users who need raw layouts (e.g., for testing Awkward internals) can use
  `contents()` directly
- New node types are added to `contents/` without touching `constructors/`

### 2. Wrappers Pattern

**Decision:** Nesting strategies (`regular_array_contents`,
`list_offset_array_contents`, `list_array_contents`) accept
`Content | Strategy | None` and compose via `contents()` drawing depth and
randomly choosing wrappers.

**Rationale:**

- `None` enables standalone use (wrapper draws its own content recursively)
- `Strategy` enables composition (e.g., `contents()` passes `st.just(content)`)
- Concrete `Content` enables deterministic wrapping in tests
- `contents()` implements the composition loop: draw a leaf, draw a depth, draw
  wrapper functions, apply in reverse order

### 3. Polymorphic `content` Parameter (Three-Form Dispatch)

**Decision:** Wrapper strategies use `match`/`case` to dispatch on the
`content` parameter type.

```python
match content:
    case None:
        content = draw(st_ak.contents.contents())
    case st.SearchStrategy():
        content = draw(content)
    case Content():
        pass
```

**Rationale:**

- Three forms cover all use cases: auto-generate, draw from strategy, use
  as-is
- Pattern matching makes the dispatch explicit and readable
- Follows the same convention as `numpy_forms(type_=...)` which accepts
  `NumpyType | Strategy | None`

### 4. `content` Is Positional

**Decision:** `content` is a positional parameter in wrapper strategies, per the
[positional-keyword-convention](../notes/2026-02-12-positional-keyword-convention.md)
(Group A).

**Rationale:**

- `content` is the "what" parameter — the primary input the wrapper operates on
- Mirrors Hypothesis `lists(elements, *, ...)` where the subject is positional
- No keyword-only parameters exist yet, but `*` future-proofs the signature for
  when config params (e.g., `max_size`) are added

### 5. `leaf_contents()` Uses `*` (All Keyword-Only)

**Decision:** `leaf_contents()` uses `*` before all parameters.

**Rationale:**

- `leaf_contents()` has no "what" parameter — it is all configuration
- Falls in Group B of the positional-keyword convention
- Matches `contents()`, `numpy_array_contents()`, and similar all-config
  strategies

### 6. Scalar Budget via CountdownDrawer

**Decision:** `max_size` controls total scalars across all leaves, managed by
`CountdownDrawer`.

**Rationale:**

- In nested structures, controlling only the outermost dimension does not bound
  total array size
- `CountdownDrawer` maintains a running total and returns `None` when the
  budget is exhausted, ensuring the strategy terminates
- Per-leaf `min_size` is managed internally — the drawer raises effective
  minimums to satisfy total constraints
- `contents()` does not expose `min_size` to keep the public API simple

### 7. Leaf Type Gating with `min_size` Guard

**Decision:** `EmptyArray` is only included as a leaf option when
`min_size == 0`.

**Rationale:**

- `EmptyArray` always has length 0, so it cannot satisfy a `min_size > 0`
  constraint
- The guard is in `leaf_contents()`, which is called by both `contents()` (via
  `CountdownDrawer`) and directly by users
- At least one leaf type must be enabled; disabling both raises `ValueError`

### 8. Global Constants

**Decision:** Use module-level constants for wrapper size limits.

| Constant           | Value | Location                                         |
| ------------------ | ----- | ------------------------------------------------ |
| `MAX_REGULAR_SIZE` | `5`   | `contents/regular_array.py`                      |
| `MAX_LIST_LENGTH`  | `5`   | `contents/list_offset_array.py`, `list_array.py` |

**Rationale:**

- Keeps generated arrays small and test-friendly
- Not exposed as strategy parameters — advanced users can compose custom
  wrappers
- Can be promoted to parameters later if needed

## Relationship to Existing Strategies

```text
Existing:
  supported_dtypes()  -->  numpy_arrays()   -->  from_numpy()  --> ak.Array
                       -->  numpy_types()                        (via ak.from_numpy)
                       -->  numpy_forms()

Contents layer:
  supported_dtypes()  -->  numpy_array_contents()  -->  leaf_contents()
                                                        empty_array_contents()

  leaf_contents()  -->  contents()  ─┬─>  regular_array_contents()
                                     ├─>  list_offset_array_contents()
                                     └─>  list_array_contents()

Constructors layer:
  contents()  -->  arrays()  -->  ak.Array
```

### Module Structure

```text
src/hypothesis_awkward/strategies/
├── contents/
│   ├── __init__.py           # Re-exports all 7 strategies
│   ├── content.py            # contents() — top-level recursive composition
│   ├── leaf.py               # leaf_contents() — leaf node selector
│   ├── numpy_array.py        # numpy_array_contents()
│   ├── empty_array.py        # empty_array_contents()
│   ├── regular_array.py      # regular_array_contents()
│   ├── list_offset_array.py  # list_offset_array_contents()
│   └── list_array.py         # list_array_contents()
├── constructors/
│   ├── __init__.py           # Re-exports arrays()
│   └── array_.py             # arrays() — thin wrapper around contents()
```

## Usage Examples

### Generate Any Content Layout

```python
import hypothesis_awkward.strategies as st_ak
from hypothesis import given

@given(c=st_ak.contents.contents())
def test_content_is_valid(c):
    assert isinstance(c, ak.contents.Content)
```

### Flat Leaf Content Only

```python
@given(c=st_ak.contents.contents(
    allow_regular=False, allow_list_offset=False, allow_list=False,
))
def test_flat_content(c):
    assert isinstance(c, (ak.contents.NumpyArray, ak.contents.EmptyArray))
```

### Wrap Specific Content in RegularArray

```python
from hypothesis import strategies as st

leaf = st_ak.contents.numpy_array_contents(max_size=20)

@given(c=st_ak.contents.regular_array_contents(leaf))
def test_regular_wrapping(c):
    assert isinstance(c, ak.contents.RegularArray)
```

### Deterministic Wrapping

```python
import awkward as ak
import numpy as np

inner = ak.contents.NumpyArray(np.array([1, 2, 3, 4]))

@given(c=st_ak.contents.regular_array_contents(inner))
def test_wrap_known_content(c):
    assert c.content is inner
```

### Control Scalar Budget

```python
from hypothesis_awkward.util import iter_numpy_arrays

@given(c=st_ak.contents.contents(max_size=50))
def test_bounded_scalars(c):
    total = sum(arr.size for arr in iter_numpy_arrays(c))
    assert total <= 50
```

### Only NumpyArray Leaves (No EmptyArray)

```python
@given(c=st_ak.contents.contents(allow_empty=False))
def test_no_empty(c):
    for leaf in iter_numpy_arrays(c):
        assert isinstance(leaf, np.ndarray)
```

## Testing Plan

Following the patterns in
[testing-patterns.md](./../../.claude/rules/testing-patterns.md):

### `contents()` tests (`tests/strategies/contents/test_content.py`)

- **TypedDict**: `ContentsKwargs` mirrors all parameters
- **Kwargs strategy**: `contents_kwargs()` with `OptsChain` for `dtypes`
  (strategy-valued kwarg)
- **Main property test**: Verifies all options — `max_size` (total scalars),
  per-type gating (`allow_regular`, `allow_list_offset`, `allow_list`), dtypes
  via leaf arrays, `allow_nan`, `max_depth`, `allow_numpy`/`allow_empty`
  `ValueError` guard
- **Edge case reachability tests** (`find()`):
  - `test_draw_max_size` — can produce content with exactly `max_size` scalars
  - `test_draw_nan` — can produce NaN values when `allow_nan=True`
  - `test_draw_integer_dtype` — can produce integer-typed content
  - `test_draw_max_depth` — can reach exactly `max_depth` nesting
  - `test_draw_nested` — can produce depth >= 2

### Individual strategy tests

Each wrapper and leaf strategy has its own test file:

```text
tests/strategies/contents/
├── test_content.py           # contents()
├── test_leaf.py              # leaf_contents()
├── test_numpy_array.py       # numpy_array_contents()
├── test_empty_array.py       # empty_array_contents()
├── test_regular_array.py     # regular_array_contents()
├── test_list_offset_array.py # list_offset_array_contents()
└── test_list_array.py        # list_array_contents()
```

## Alternatives Considered

### Alternative A: Single `contents()` Function With All Logic Inline

Place all content generation logic in a single function rather than splitting
into separate modules.

**Rejected because:**

- Each node type has distinct construction logic (divisor selection for
  `RegularArray`, offset generation for `ListOffsetArray`, starts/stops for
  `ListArray`)
- Separate modules enable independent testing and clear ownership
- New node types are added as new files, not modifications to a monolith

### Alternative B: Expose `min_size` on `contents()`

Allow users to set a minimum total scalar count.

**Deferred because:**

- `CountdownDrawer` already manages per-leaf minimums internally
- The interaction between `min_size` and nesting depth is complex (e.g., nested
  wrappers can multiply the effective minimum)
- Can be added later if there is demand

### Alternative C: Expose `MAX_REGULAR_SIZE` / `MAX_LIST_LENGTH` as Parameters

Let users control the maximum wrapper size via strategy parameters.

**Deferred because:**

- Keeps the public API simple for the initial version
- Users who need different limits can compose custom wrappers
- Can be promoted to parameters in a future iteration

### Alternative D: Wrapper Strategies Accept Only `Strategy`, Not `Content`

Require all content to be passed as strategies, using `st.just()` for concrete
values.

**Rejected because:**

- Accepting concrete `Content` directly is more ergonomic for testing
- The three-form dispatch (`None` / `Strategy` / `Content`) covers all use
  cases naturally
- Pattern matching makes the dispatch clean

## Open Questions

1. **Should wrapper strategies expose size parameters?** Currently,
   `MAX_REGULAR_SIZE` and `MAX_LIST_LENGTH` are module-level constants. If
   promoted to parameters, the wrappers would gain keyword-only config after
   `*`, which the signature already accommodates.

2. **Should `contents()` support `min_size`?** Users may want to guarantee a
   minimum number of scalars. This would require `CountdownDrawer` to expose
   minimum guarantees to the public API.

3. **How will `RecordArray` and option types affect the wrappers pattern?**
   Records have multiple contents (one per field), not a single `content`.
   Option types add masking layers. The composition loop in `contents()` may
   need constraint tracking (e.g., option nodes cannot wrap other option nodes).

## Completed

1. ~~Implement `numpy_array_contents()`~~ ✓
2. ~~Implement `empty_array_contents()`~~ ✓
3. ~~Implement `leaf_contents()` with type gating~~ ✓
4. ~~Implement `regular_array_contents()` with divisor selection~~ ✓
5. ~~Implement `list_offset_array_contents()` with offset generation~~ ✓
6. ~~Implement `list_array_contents()` with starts/stops~~ ✓
7. ~~Implement `contents()` with wrappers pattern and `CountdownDrawer`~~ ✓
8. ~~Write tests for all strategies~~ ✓
9. ~~Export from `contents/__init__.py`~~ ✓

## Next Steps

1. Add `RecordArray` support (`record_array_contents()`)
2. Add option type support (`indexed_option_array_contents()`,
   `byte_masked_array_contents()`, etc.)
3. Add `UnionArray` support (`union_array_contents()`)
4. Add string/bytestring support
5. Consider exposing `MAX_REGULAR_SIZE` / `MAX_LIST_LENGTH` as parameters
6. Consider adding `min_size` to `contents()`
