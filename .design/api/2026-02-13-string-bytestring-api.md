# API Design: String and Bytestring Content Strategies

**Date:** 2026-02-13
**Status:** Proposed
**Author:** Claude (with developer collaboration)

## Overview

This document describes the API for `string_contents()` and
`bytestring_contents()` strategies and their integration into the existing
`contents()` / `arrays()` pipeline.

See [string-bytestring-research](../research/2026-02-12-string-bytestring-research.md)
for background on Awkward Array's string representation.

## Background

Strings and bytestrings in Awkward Array are **not** a separate Content class.
They are `ListOffsetArray` (or `ListArray` / `RegularArray`) nodes with
`__array__` parameters wrapping a `NumpyArray(uint8)` with a matching
`__array__` parameter:

```text
ListOffsetArray(parameters={"__array__": "string"})
  └── NumpyArray(dtype=uint8, parameters={"__array__": "char"})
```

This structure is **leaf-like**: the inner content is always a fixed
`NumpyArray(uint8)`, never arbitrary content. String nodes cannot wrap other
content types. This places them in the same category as `numpy_array_contents()`
and `empty_array_contents()` — they are terminal nodes in the content tree.

## Design Goals

1. **Leaf-like treatment**: String/bytestring strategies produce complete,
   self-contained Content nodes (list + inner NumpyArray). They participate in
   `leaf_contents()` selection alongside `NumpyArray` and `EmptyArray`.
2. **Valid-by-construction**: Strings produce valid UTF-8 via `st.text()`.
   Bytestrings produce arbitrary bytes via `st.binary()`.
3. **Consistent integration**: New `allow_string` / `allow_bytestring` flags on
   `contents()` and `arrays()` follow the established `allow_*` pattern.
4. **Scalar budgeting**: String bytes count toward the `max_size` scalar budget,
   consistent with how `NumpyArray` scalars are counted.

## API

### `string_contents()`

Generates a `ListOffsetArray` of UTF-8 strings with `__array__` parameters.

```python
@st.composite
def string_contents(
    draw: st.DrawFn,
    *,
    alphabet: st.SearchStrategy[str] | None = None,
    min_size: int = 0,
    max_size: int = 10,
) -> ListOffsetArray:
```

#### Parameters

- **`alphabet`** — Strategy for characters used in generated strings. If `None`
  (default), uses `st.characters()` (Hypothesis default for `st.text()`). Passed
  directly to `st.text(alphabet=...)`.

- **`min_size`** — Minimum number of strings (list elements). Default: `0`.
  This is the minimum number of strings in the array, not the minimum byte
  count. A `min_size=1` array may contain a single empty string `""`.

- **`max_size`** — Maximum number of strings (list elements). Default: `10`.
  This bounds the number of strings, not the total byte count. Individual string
  lengths are drawn independently.

#### Return Type

`ListOffsetArray` with `parameters={"__array__": "string"}`, wrapping a
`NumpyArray(uint8)` with `parameters={"__array__": "char"}`.

#### Behavior

1. Draw `n` strings from `st.lists(st.text(alphabet=alphabet), min_size=...,
   max_size=...)`.
2. Encode each string to UTF-8 bytes, concatenate into a single `uint8` buffer.
3. Compute offsets from cumulative byte lengths.
4. Construct `NumpyArray(buffer, parameters={"__array__": "char"})`.
5. Construct `ListOffsetArray(offsets, content,
   parameters={"__array__": "string"})`.

### `bytestring_contents()`

Generates a `ListOffsetArray` of bytestrings with `__array__` parameters.

```python
@st.composite
def bytestring_contents(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 10,
) -> ListOffsetArray:
```

#### Parameters

- **`min_size`** — Minimum number of bytestrings (list elements). Default: `0`.

- **`max_size`** — Maximum number of bytestrings (list elements). Default: `10`.

#### Return Type

`ListOffsetArray` with `parameters={"__array__": "bytestring"}`, wrapping a
`NumpyArray(uint8)` with `parameters={"__array__": "byte"}`.

#### Behavior

Same as `string_contents()` but uses `st.binary()` instead of `st.text()`. No
`alphabet` parameter since all byte values are valid.

### Changes to `leaf_contents()`

Add `allow_string` and `allow_bytestring` parameters:

```python
def leaf_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = False,
    allow_bytestring: bool = False,
) -> st.SearchStrategy[NumpyArray | EmptyArray | ListOffsetArray]:
```

When `allow_string=True`, includes `string_contents(min_size=...,
max_size=...)` in the `st.one_of()` options. Same for `allow_bytestring`.

The `min_size` guard applies: string/bytestring options are only included when
they can satisfy the minimum. Since `string_contents(min_size=0)` can produce a
zero-length array (no strings), they are excluded when `min_size > 0` requires
at least one scalar — unless `min_size` refers to list elements (see Design
Decision 3).

The validation rule changes to: at least one of `allow_numpy`, `allow_empty`,
`allow_string`, or `allow_bytestring` must be `True`.

### Changes to `contents()`

Add `allow_string` and `allow_bytestring` parameters:

```python
@st.composite
def contents(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = False,
    allow_bytestring: bool = False,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> Content:
```

These are forwarded to `leaf_contents()` via `functools.partial`.

### Changes to `arrays()`

Same two new parameters, forwarded to `contents()`.

## Design Decisions

### 1. Leaf-Like, Not Wrapper

**Decision:** String/bytestring strategies are leaf-like nodes in the content
tree, not wrapper strategies.

**Rationale:**

- String content is always `NumpyArray(uint8)` — it cannot wrap arbitrary
  content.
- String nodes are terminal: nesting layers go _above_ them, not _inside_ them.
- This matches how `contents()` works: draw a leaf, then wrap it in nesting
  layers. A string leaf can be wrapped in `RegularArray` or `ListOffsetArray`
  to produce nested string arrays (list-of-strings).

**Alternative considered:** Treating strings as a special case of
`list_offset_array_contents()` with a `parameters` argument. Rejected because
the inner content is fixed (`NumpyArray(uint8)`) — the wrapper pattern of
accepting arbitrary content does not apply.

### 2. Default `allow_string=False` / `allow_bytestring=False`

**Decision:** String and bytestring generation is opt-in.

**Rationale:**

- Adding strings to the default leaf pool would change the output distribution
  of existing `contents()` / `arrays()` calls, potentially breaking existing
  tests.
- Strings exercise different code paths than numeric arrays. Users should
  opt in when testing string-handling code.
- This matches the incremental extensibility goal: new node types are added as
  new `allow_*` flags without changing defaults.

**Alternative considered:** Defaulting to `True`. Deferred until the strategy
is stable and widely tested. Can change the default in a future version.

### 3. `min_size` / `max_size` Count Strings, Not Bytes

**Decision:** The `min_size` and `max_size` parameters on `string_contents()`
and `bytestring_contents()` count the number of strings (list elements), not
the total byte count.

**Rationale:**

- Consistent with `list_offset_array_contents()` where size refers to the
  number of sublists, not the total content length.
- Users think in terms of "how many strings" not "how many bytes".
- Individual string lengths are controlled by Hypothesis's `st.text()` /
  `st.binary()` defaults, which produce reasonably short strings.

### 4. `alphabet` Parameter on `string_contents()` Only

**Decision:** Only `string_contents()` has an `alphabet` parameter.
`bytestring_contents()` does not.

**Rationale:**

- For strings, `alphabet` controls which characters appear. This is useful for
  testing ASCII-only code, specific character sets, etc.
- For bytestrings, all 256 byte values are valid. There is no meaningful
  "alphabet" to restrict. Users who need specific byte patterns can compose
  their own strategy.

### 5. `ListOffsetArray` Only (Not `ListArray` or `RegularArray`)

**Decision:** `string_contents()` and `bytestring_contents()` produce
`ListOffsetArray` only. They do not generate `ListArray` or `RegularArray`
string layouts.

**Rationale:**

- `ListOffsetArray` is the canonical representation — it is what `ak.Array(
  ["..."])` produces and what most Awkward operations return.
- `ListArray` and `RegularArray` string layouts are rare in practice (see
  research doc: "Variable-length strings (dominant)").
- Keeping the initial implementation simple. `ListArray` and `RegularArray`
  variants can be added later if needed, potentially as separate strategies or
  via a `list_type` parameter.

### 6. No `alphabet` / `bytestring_alphabet` on `contents()` / `arrays()`

**Decision:** The top-level `contents()` and `arrays()` strategies do not
expose `alphabet` parameters. They forward only `allow_string` and
`allow_bytestring` to `leaf_contents()`.

**Rationale:**

- `contents()` already has many parameters. Adding `alphabet` would add
  complexity for a niche use case.
- Users who need alphabet control can use `string_contents(alphabet=...)`
  directly and pass it as a concrete leaf.
- Consistent with how `contents()` does not expose per-leaf configuration
  beyond `dtypes` and `allow_nan`.

### 7. Scalar Budget: Bytes Count as Scalars

**Decision:** When `string_contents()` / `bytestring_contents()` participate
in the `CountdownDrawer` scalar budget, the total byte count of all strings
counts toward `max_size`.

**Rationale:**

- The inner `NumpyArray(uint8)` is the actual data storage. Its `.size` is the
  total byte count.
- `iter_numpy_arrays()` already visits this inner array, so existing budget
  accounting works without changes.
- Consistent with how `numpy_array_contents()` scalar counts work.

### 8. Group B (All Keyword-Only)

**Decision:** Both `string_contents()` and `bytestring_contents()` use
keyword-only parameters (Group B in the positional-keyword convention).

**Rationale:**

- Neither strategy has a "what" parameter — everything is configuration.
- Matches `numpy_array_contents()`, `leaf_contents()`, `contents()`, etc.

## Module Structure

### New Files

```text
src/hypothesis_awkward/strategies/contents/
├── ...existing files...
├── string.py              # string_contents()
└── bytestring.py          # bytestring_contents()
```

### Exports

Add to `contents/__init__.py`:

```python
__all__ = [
    ...existing...,
    'string_contents',
    'bytestring_contents',
]
```

## Relationship Diagram

```text
Leaf strategies:
  numpy_array_contents()   ──┐
  empty_array_contents()   ──┤
  string_contents()        ──┼──>  leaf_contents()  ──>  contents()  ──>  arrays()
  bytestring_contents()    ──┘

Nesting strategies (wrap any leaf, including strings):
  regular_array_contents()       ──┐
  list_offset_array_contents()   ──┼──>  contents()
  list_array_contents()          ──┘
```

A string leaf wrapped in a nesting layer produces nested string arrays:

```text
contents(allow_string=True, max_depth=1)  may produce:

ListOffsetArray (no __array__)              -- list-of-strings
  ListOffsetArray (__array__="string")      -- string leaf
    NumpyArray uint8 (__array__="char")
```

## Usage Examples

### Basic String Content

```python
import hypothesis_awkward.strategies as st_ak
from hypothesis import given

@given(c=st_ak.contents.string_contents())
def test_string_content(c):
    a = ak.Array(c)
    assert a.type.content == ak.types.NumpyType('uint8', parameters={'__array__': 'char'})
```

### ASCII-Only Strings

```python
from hypothesis import strategies as st

@given(c=st_ak.contents.string_contents(alphabet=st.characters(codec='ascii')))
def test_ascii_strings(c):
    for s in ak.Array(c).to_list():
        assert s.isascii()
```

### Mixed Content with Strings

```python
@given(a=st_ak.constructors.arrays(allow_string=True))
def test_string_handling(a):
    # a may contain string arrays alongside numeric arrays
    ...
```

### Nested Strings (List of Strings)

```python
@given(a=st_ak.constructors.arrays(allow_string=True, max_depth=2))
def test_nested_strings(a):
    # may produce list-of-list-of-strings
    ...
```

## Testing Plan

Following the patterns in
[testing-patterns.md](../../.claude/rules/testing-patterns.md):

### `string_contents()` tests (`tests/strategies/contents/test_string.py`)

- **TypedDict**: `StringContentsKwargs` with `alphabet`, `min_size`, `max_size`
- **Kwargs strategy**: `string_contents_kwargs()` with `OptsChain` for
  `alphabet` (strategy-valued kwarg)
- **Main property test**: Verifies `__array__` parameters, UTF-8 validity,
  size bounds, dtype is `uint8`
- **Edge case reachability tests** (`find()`):
  - `test_draw_empty` — can produce zero-length string array
  - `test_draw_non_ascii` — can produce non-ASCII strings
  - `test_draw_empty_string` — can produce array containing empty string `""`
  - `test_draw_from_contents` — string content can be drawn via
    `contents(allow_string=True)`

### `bytestring_contents()` tests (`tests/strategies/contents/test_bytestring.py`)

- **TypedDict**: `BytestringContentsKwargs` with `min_size`, `max_size`
- **Main property test**: Verifies `__array__` parameters, size bounds, dtype
  is `uint8`
- **Edge case reachability tests** (`find()`):
  - `test_draw_empty` — can produce zero-length bytestring array
  - `test_draw_from_contents` — bytestring content can be drawn via
    `contents(allow_bytestring=True)`

### Updates to existing tests

- `test_content.py`: Add tests for `allow_string` and `allow_bytestring` flags
  in the main property test

## Open Questions

1. **Should individual string length be configurable?** Currently, individual
   string lengths are determined by `st.text()` defaults. A `max_str_length`
   parameter could be added later if needed.

2. **Should `ListArray` and `RegularArray` string variants be supported?** The
   initial implementation uses `ListOffsetArray` only. Separate strategies or a
   `list_type` parameter could be added later.

3. **How does `max_size` interact with the scalar budget for strings?** Since
   bytes count as scalars, a `max_size=10` budget could produce at most 10 bytes
   total across all strings. This may be too restrictive. The `max_size` on
   `string_contents()` could instead control the number of strings, with byte
   count uncapped. This needs to be resolved during implementation.
