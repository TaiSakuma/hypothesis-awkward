# API Design: `max_length` Parameter

**Date:** 2026-02-23
**Status:** In progress
**Author:** Claude (with developer collaboration)

## Overview

This document describes the API for adding a `max_length` parameter to the
`contents/` strategies. `max_length` caps the immediate `len()` of generated
content at a single level, independent of `max_size` (total scalar budget across
all leaves).

See [max-length-research](../research/2026-02-23-max-length-research.md) for
background on the two length concepts and motivation.

## Background

The codebase currently uses `max_size` to bound total scalars across all leaf
nodes. There is no way to constrain the immediate `len()` of a content node
independently:

- **`max_size`** — Total scalar count across all `NumpyArray` leaves, strings,
  and bytestrings. A `RegularArray` of size 3 wrapping a `NumpyArray` of
  length 12 has `max_size` cost of 12.

- **`max_length`** (new) — Immediate `len()` of a content node at one level.
  For the same `RegularArray`, `max_length` constrains `len(content) // size`,
  i.e., the number of groups, not the total elements.

Both constraints hold simultaneously: a content node must have
`len() <= max_length` **and** total leaf scalars `<= max_size`.

The primary motivation is to improve `size` coverage in
`regular_array_contents()`. With `max_length` on child content, the strategy can
draw `size` first and then request content whose length is a multiple of `size`,
giving uniform coverage over `size` values instead of being biased by the
divisor structure of whatever content length happens to be drawn.

## Per-Strategy Semantics

### Wrapper strategies

These strategies receive `max_length` from their callers and use it to constrain
the immediate `len()` of their output.

#### `regular_array_contents()`

`max_length` constrains the number of groups (the array length after dividing
by `size`):

```text
len(content) // size <= max_length
```

This means child content length is bounded by `max_length * size`. When
`max_length` is provided, the strategy can draw `size` first and then request
child content with length that is a multiple of `size`, up to
`max_length * size`.

#### `list_offset_array_contents()`

`max_length` constrains the number of sublists:

```text
len(offsets) - 1 <= max_length
```

The number of lists is drawn from `[0, max_length]`, defaulting to `5`.

#### `list_array_contents()`

Same semantics as `list_offset_array_contents()`:

```text
len(starts) <= max_length
```

The number of lists is drawn from `[0, max_length]`, defaulting to `5`.

#### `record_array_contents()`

`max_length` constrains the shared field length (all fields in a `RecordArray`
have the same `len()`):

```text
length <= max_length
```

The RecordArray's `length` is capped to `min(min(len(c) for c in contents),
max_length)`. Default: `None` (no constraint).

#### `union_array_contents()`

`max_length` constrains the union length:

```text
len(result) <= max_length
```

When `contents` is `None`, `content_lists(max_total_size=max_length)` steers
generation and `assume()` filters any overshoot. When concrete or
strategy-valued `contents` are provided, the strategy builds compact tags/index,
shuffles them, and truncates to `max_length` entries. This produces non-compact
indexing — some child elements become unreachable — but avoids `Unsatisfied`
when the caller provides contents whose total length exceeds `max_length`.
Default: `None` (no constraint).

### Leaf strategies

For leaf strategies, `max_length` and `max_size` both constrain the same
dimension (immediate length equals total scalars for flat arrays). The effective
limit is `min(max_size, max_length)`.

#### `numpy_array_contents()`

```text
len(array) <= min(max_size, max_length)
```

The effective `max_size` passed to the underlying `numpy_arrays()` strategy is
`min(max_size, max_length)`.

#### `string_contents()`

```text
number of strings <= min(max_size, max_length)
```

The effective `max_size` passed to `st.lists(st.text(...), max_size=...)` is
`min(max_size, max_length)`.

#### `bytestring_contents()`

```text
number of bytestrings <= min(max_size, max_length)
```

Same treatment as `string_contents()`.

### Selector and entry-point strategies

#### `leaf_contents()`

Accepts `max_length` and passes it through to each leaf sub-strategy. No
additional logic beyond forwarding.

#### `content_lists()`

Accepts `max_length` and propagates it to each child content drawn. Each child
in the list independently satisfies the `max_length` constraint.

#### `contents()`

New `max_length` parameter wired to all child strategies. `contents()` passes
`max_length` to whichever wrapper or leaf strategy it selects. The parameter
is orthogonal to `max_size` and `max_depth`.

## Parameter Design

### Name

`max_length` — consistent with Hypothesis naming conventions (e.g.,
`st.lists(max_size=...)` uses "size" for element count; we use "length" to
distinguish from the existing `max_size` which means total scalars in this
codebase).

### Type

`int` — same as `max_size`.

### Default

`None` — when not specified, no immediate-length constraint is applied. This
preserves backward compatibility: existing behavior is unchanged when
`max_length` is not passed.

### Position

Keyword-only in all strategies. Placed after `max_size` where both exist:

```python
def numpy_array_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
    max_length: int | None = None,
) -> st.SearchStrategy[NumpyArray]:
```

For wrapper strategies with a positional `content` parameter:

```python
@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int = 5,
    max_zeros_length: int = 5,
    max_length: int | None = None,
) -> Content:
```

For `contents()`:

```python
@st.composite
def contents(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    max_length: int | None = None,
    allow_nan: bool = False,
    # ... remaining parameters unchanged
) -> Content:
```

## Interaction with `max_size`

Both constraints hold simultaneously. Neither subsumes the other:

| Scenario              | `max_size`   | `max_length` | Effect                                  |
| --------------------- | ------------ | ------------ | --------------------------------------- |
| Only `max_size` set   | 10           | `None`       | Total scalars <= 10, no length cap      |
| Only `max_length` set | 10 (default) | 5            | Total scalars <= 10, immediate len <= 5 |
| Both set              | 20           | 3            | Total scalars <= 20, immediate len <= 3 |
| Leaf node             | 10           | 5            | Effective limit: `min(10, 5) = 5`       |

For wrapper strategies, the constraints operate on different dimensions:

- A `RegularArray` with `max_size=20, max_length=3` can have at most 3 groups,
  but child content can have up to 20 total scalars (e.g., 3 groups of size 6
  = 18 elements).
- A `ListOffsetArray` with `max_size=20, max_length=3` can have at most 3
  sublists, but child content can have up to 20 total scalars.

## Design Decisions

### 1. `None` Default (Opt-In)

**Decision:** `max_length` defaults to `None`, meaning no constraint.

**Rationale:**

- Backward compatibility: existing tests and usage are unaffected.
- `max_length` is primarily needed by `regular_array_contents()` internally for
  `size` coverage improvement. Most users do not need to set it directly.
- `None` is idiomatic for "no limit" in Hypothesis (e.g., `max_examples`).

### 2. `int | None` Type (Not `int` With Sentinel)

**Decision:** Use `int | None` rather than a sentinel integer like `sys.maxsize`.

**Rationale:**

- Explicit `None` is clearer than a magic large number.
- Avoids surprising behavior when `max_length` is accidentally set to
  `sys.maxsize` and interacts with arithmetic.
- Consistent with how optional parameters work elsewhere in the codebase.

### 3. Leaf Strategies Use `min(max_size, max_length)`

**Decision:** For leaf strategies where immediate length equals total scalars,
the effective limit is `min(max_size, max_length)` (treating `None` as
unbounded).

**Rationale:**

- Both constraints must hold. For flat arrays, both refer to the same dimension,
  so the tighter constraint wins.
- Keeps the implementation simple: compute effective max, pass to underlying
  strategy.

### 4. Wrapper Strategies Constrain Output Length

**Decision:** Each wrapper strategy interprets `max_length` as a constraint on
its own output `len()`, not on its child content's `len()`.

**Rationale:**

- The caller wants to control the length of the result it receives.
- Wrapper strategies may internally request child content of different length
  (e.g., `regular_array_contents` needs child length = `size * output_length`).
- This keeps the semantics uniform: `max_length` always means "the thing you
  get back has `len()` at most this."

### 5. `union_array_contents()` Uses Non-Compact Indexing for Truncation

**Decision:** For `union_array_contents()`, `max_length` constrains
`len(result)`. When concrete or strategy-valued contents exceed `max_length`,
the shuffled tags/index arrays are truncated rather than rejected via
`assume()`.

**Rationale:**

- `assume()` on concrete/strategy contents causes `Unsatisfied` when the total
  content length always exceeds `max_length`.
- `UnionArray` does not require compact indexing — tags/index can reference a
  subset of each child's elements, leaving some unreachable.
- Truncation after shuffling gives a uniform random subset of the interleaved
  elements.
- `assume()` is still used in the `None` branch where `content_lists()` steers
  generation and overshoot is rare.

## Implementation Order

See [max-length-research](../research/2026-02-23-max-length-research.md) for
the bottom-up implementation order. Wrapper strategies are implemented first
(they receive `max_length`), then leaves, then the entry points that wire
everything together.

### Progress

1. `regular_array_contents()` — done
2. `list_offset_array_contents()` — done
3. `list_array_contents()` — done
4. `record_array_contents()` — done
5. `union_array_contents()` — done
6. `numpy_array_contents()` — N/A (`max_size` is equivalent)
7. `string_contents()` — N/A (`max_size` is equivalent)
8. `bytestring_contents()` — N/A (`max_size` is equivalent)
9. `leaf_contents()` — N/A (forwards to leaf strategies via `max_size`)
