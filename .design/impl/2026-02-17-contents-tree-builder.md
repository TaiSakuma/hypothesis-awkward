# Implementation: Bottom-Up Tree Builder for `contents()`

**Date:** 2026-02-17
**Status:** Implemented
**Author:** Claude (with developer collaboration)

## Overview

This document describes the refactoring of the `contents()` generation
algorithm from a linear wrapper chain to a recursive bottom-up tree builder.
This enables `RecordArray` support (and future `UnionArray`) by allowing
multi-child nodes in the content tree.

See [record-array-research](../research/2026-02-17-record-array-research.md)
for background on `RecordArray`.

## Motivation

The previous `contents()` built content as a linear chain:

```text
leaf → wrapper₁ → wrapper₂ → ... → wrapperₙ
```

Each wrapper has exactly one child. This cannot accommodate `RecordArray`
(0+ children) or `UnionArray` (2+ children).

## Algorithm

### Previous: Linear Chain

```python
draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)
content = draw_leaf()
depth = draw(st.integers(0, max_depth))
nesting = [draw(st.sampled_from(nesting_fns)) for _ in range(depth)]
for fn in reversed(nesting):
    content = draw(fn(st.just(content)))
return content
```

### Current: Bottom-Up Tree Builder

A single recursive function `_build(depth)` that:

1. **Going down**: draws "deeper?" or "bottom?" — nothing else.
2. **At the bottom**: draws a leaf instance.
3. **Going up**: draws "another edge?".
   - If no: draws a node type that can accommodate the number of children.
   - If yes: goes down that edge (recursive call), then asks "another edge?"
     again.

### Walkthrough

```text
 1. Going down — deeper? Yes
 2. Going down — deeper? Yes
 3. Going down — deeper? No (bottom)
 4. Draw leaf: NumpyArray([1, 2, 3])
 5. Go up — another edge? No
    1 child → draw from {Regular, ListOffset, List, RecordArray}
    Draw: RegularArray. Construct it.
 6. Go up — another edge? Yes
    Go down:
      7. Going down — deeper? No (bottom)
      8. Draw leaf: NumpyArray([4.0, 5.0])
    Another edge? Yes
    Go down:
      9.  Going down — deeper? Yes
      10. Going down — deeper? No (bottom)
      11. Draw leaf: string(["hello", "world"])
      12. Go up — another edge? No
          1 child → draw: ListOffsetArray. Construct it.
    Another edge? No
    3 children → draw from {RecordArray}
    Draw: named RecordArray. Construct it.
13. At the top. Done.
```

Result:

```text
RecordArray (named, fields=['f0', 'f1', 'f2'])
├── f0: RegularArray
│       └── NumpyArray([1, 2, 3])
├── f1: NumpyArray([4.0, 5.0])
└── f2: ListOffsetArray
        └── string(["hello", "world"])
```

### Node Type Selection

The node type is drawn **after** all children are collected, constrained by the
child count:

| Children | Possible node types                                   |
| -------- | ----------------------------------------------------- |
| 1        | RegularArray, ListOffsetArray, ListArray, RecordArray |
| 2+       | RecordArray (and UnionArray in the future)            |

"Another edge?" is only offered when a multi-child node type is enabled. If
`allow_record=False` (and no `UnionArray`), "another edge?" is always no.

### RecordArray Construction

When `RecordArray` is selected:

- **Named vs. tuple**: drawn (e.g., `draw(st.booleans())`)
- **Field names**: for named records, generated as `'f0'`, `'f1'`, ...
- **Length**: `min(len(c) for c in children)`

### Depth Counting

`RecordArray` consumes a depth level, same as list wrappers. A
`RecordArray → ListOffset → leaf` path has depth 2.

### Budget Management

The `CountdownDrawer` is shared across all leaves in the tree. Leaves are
generated in DFS order (leftmost branch first). Earlier branches may consume
more budget than later ones. This is acceptable — Hypothesis explores many
random orderings across test runs.

## Pseudocode

```python
@st.composite
def contents(draw, *, ..., allow_record=True) -> Content:

    st_leaf = functools.partial(leaf_contents, ...)

    wrappers: dict[str, _WrapperFn] = {}
    if allow_regular:
        wrappers['regular'] = st_ak.contents.regular_array_contents
    if allow_list_offset:
        wrappers['list_offset'] = st_ak.contents.list_offset_array_contents
    if allow_list:
        wrappers['list'] = st_ak.contents.list_array_contents

    can_branch = allow_record  # (or allow_union in the future)

    single_child_types = list(wrappers)
    if allow_record:
        single_child_types.append('record')

    if not single_child_types or max_size == 0:
        return draw(st_leaf(min_size=0, max_size=max_size))

    draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)

    def _leaf() -> Content:
        content = draw_leaf()
        if content is not None:
            return content
        return draw(st_leaf(min_size=0, max_size=0))

    def _build(depth: int) -> Content:
        # Going down: deeper or bottom?
        if depth >= max_depth or not draw(st.booleans()):
            return _leaf()

        # Go down first edge
        children = [_build(depth + 1)]

        # Going up: another edge?
        while can_branch and draw(st.booleans()):
            children.append(_build(depth + 1))

        # Draw node type constrained by child count
        if len(children) == 1:
            node_type = draw(st.sampled_from(single_child_types))
        else:
            node_type = 'record'  # only multi-child type for now

        # Construct node
        if node_type == 'record':
            is_tuple = draw(st.booleans())
            fields = None if is_tuple else [f'f{i}' for i in range(len(children))]
            length = min(len(c) for c in children)
            return RecordArray(children, fields, length=length)

        # Single-child wrapper
        return draw(wrappers[node_type](st.just(children[0])))

    return _build(0)
```

## Key Differences from Previous Code

| Aspect      | Previous (linear chain)             | Current (bottom-up tree)           |
| ----------- | ----------------------------------- | ---------------------------------- |
| Structure   | `nesting_fns` list, apply in loop   | `_build(depth)` recursive function |
| Depth       | Drawn upfront as integer            | Emergent from "deeper?" coin flips |
| Branching   | Always 1 child                      | 1+ children via "another edge?"    |
| Node type   | Chosen per level from wrappers list | Constrained by child count         |
| RecordArray | Not supported                       | Supported at any depth             |
| Budget      | `CountdownDrawer` feeds one leaf    | `CountdownDrawer` feeds all leaves |

## Decisions

### 1. Single-Pass Bottom-Up (Not Two-Phase)

Structure decisions ("deeper?", "another edge?") and data generation (leaf
drawing, node construction) happen in one recursive traversal. No intermediate
plan/shape data structures.

### 2. "Another Edge?" Instead of Upfront Branching Factor

The number of children per node emerges from repeated boolean draws. This
produces a geometric distribution (most nodes have few children). Budget
depletion naturally discourages wide nodes.

### 3. Node Type Determined by Child Count

Node type is drawn after all children exist, constrained by the count. This
lets 1-field RecordArray coexist with list wrappers naturally. Future
UnionArray (2+ children) slots in without changing the algorithm.

### 4. DFS Budget Order

The budget is consumed in DFS order. Earlier branches may consume more than
later ones. This is acceptable for property-based testing where variety across
runs matters more than balance within a single run.

## Open Questions

1. **"Deeper?" probability**: `draw(st.booleans())` gives 50% chance per
   level. May need tuning if trees are too shallow or too deep.

2. **"Another edge?" probability**: Same 50%. Most nodes get 1-2 children.
   May need tuning if wider records are desired.

3. **Empty records (0 fields)**: The algorithm always produces >= 1 child.
   0-field records are valid but not generated. Could be a special leaf case.

4. **Backward compatibility**: The refactoring changes the depth distribution
   (upfront integer → coin flips). Existing tests should still pass, but the
   distribution of generated content shapes will differ.
