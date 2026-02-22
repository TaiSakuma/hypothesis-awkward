# Implementation: Top-Down Tree Builder for `contents()`

**Date:** 2026-02-21
**Status:** Implemented
**Supersedes:** [Bottom-up tree builder](2026-02-17-contents-tree-builder.md)

> **Note (2026-02-22):** The implementation differs from the pseudocode below:
>
> - `contents()` is self-recursive via `functools.partial` instead of using an
>   inner `_build()` function. `CountdownDrawer` is not used; budget is tracked
>   by passing `max_size` minus `_leaf_size()` to each recursive call.
> - The no-nested-union constraint uses `allow_union_root=False` (a new
>   parameter) instead of threading `allow_union=False` down to children.

## Motivation

The bottom-up builder chooses the node type **after** building children. This
creates a mismatch: children are built without knowing what parent will hold
them. Consequences:

1. **Orphaned children** — When `allow_record=False` and a child is a
   `UnionArray`, no multi-child type can hold all children. The fallback
   discards all but the first.
2. **Post-hoc constraint checking** — The no-nested-union rule is enforced by
   `isinstance(c, UnionArray)` after children are built, rather than
   structurally.
3. **Complex candidate logic** — `_candidate_node_types()` exists solely to
   reconcile child count with available node types, including a fallback path
   for the orphan case.

## Algorithm

Choose the node type **before** building children. The parent dictates what
children can be.

```text
_build(depth, allow_union):
    deeper or leaf? → leaf
    choose node type from allow_* flags
    build children appropriate for that type
    construct node
```

### Walkthrough

```text
 1. Going down — deeper? Yes
 2. Choose node type: RecordArray
 3.   Build child 0:
 4.     Going down — deeper? Yes
 5.     Choose node type: RegularArray
 6.       Build child:
 7.         Going down — deeper? No (leaf)
 8.         Draw leaf: NumpyArray([1, 2, 3])
 9.       Construct RegularArray
10.   Another child? Budget remaining, coin flip: Yes
11.   Build child 1:
12.     Going down — deeper? Yes
13.     Choose node type: ListOffsetArray
14.       Build child:
15.         Going down — deeper? No (leaf)
16.         Draw leaf: string(["hello", "world"])
17.       Construct ListOffsetArray
18.   Another child? Coin flip: No
19.   Construct RecordArray (fields=['f0', 'f1'])
20. Done.
```

Result:

```text
RecordArray (named, fields=['f0', 'f1'])
├── f0: RegularArray
│       └── NumpyArray([1, 2, 3])
└── f1: ListOffsetArray
        └── string(["hello", "world"])
```

### Node Type Selection

The candidate set depends only on the `allow_*` flags — never on child values:

| Node type   | Min children | More children?    | Child constraints   |
| ----------- | ------------ | ----------------- | ------------------- |
| regular     | 1            | no                | none                |
| list_offset | 1            | no                | none                |
| list        | 1            | no                | none                |
| record      | 1            | coin flip per add | none                |
| union       | 2            | coin flip per add | `allow_union=False` |

### No-Nested-Union Constraint

In the bottom-up builder, this was a runtime check:

```python
if allow_union and not any(isinstance(c, UnionArray) for c in children):
    candidates.append('union')
```

In the top-down builder, it is structural. When `union` is chosen, children are
built with `allow_union=False`:

```python
if node_type == 'union':
    children = [
        _build(depth + 1, allow_union=False),
        _build(depth + 1, allow_union=False),
    ]
    while not budget_exhausted and draw(st.booleans()):
        children.append(_build(depth + 1, allow_union=False))
```

A child **cannot** be a `UnionArray` because the recursive call excludes it
from the candidate set.

## Pseudocode

```python
@st.composite
def contents(draw, *, ..., allow_union=True) -> Content:

    st_leaf = functools.partial(leaf_contents, ...)

    if leaf_only:
        return draw(st_leaf(min_size=0, max_size=max_size))

    draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)

    budget_exhausted = False

    def _leaf() -> Content:
        nonlocal budget_exhausted
        content = draw_leaf()
        if content is not None:
            return content
        budget_exhausted = True
        return draw(st_leaf(min_size=0, max_size=0))

    def _build(depth: int, *, allow_union: bool = True) -> Content:
        if budget_exhausted or depth >= max_depth or not draw(st.booleans()):
            return _leaf()

        # Choose node type from allow_* flags
        candidates: list[str] = []
        if allow_regular:
            candidates.append('regular')
        if allow_list_offset:
            candidates.append('list_offset')
        if allow_list:
            candidates.append('list')
        if allow_record:
            candidates.append('record')
        if allow_union:
            candidates.append('union')

        if not candidates:
            return _leaf()

        node_type = draw(st.sampled_from(sorted(candidates)))

        # Build children for multi-child types
        if node_type == 'union':
            children = [
                _build(depth + 1, allow_union=False),
                _build(depth + 1, allow_union=False),
            ]
            while not budget_exhausted and draw(st.booleans()):
                children.append(_build(depth + 1, allow_union=False))
            return draw(st_ak.contents.union_array_contents(children))

        if node_type == 'record':
            children = [_build(depth + 1, allow_union=allow_union)]
            while not budget_exhausted and draw(st.booleans()):
                children.append(_build(depth + 1, allow_union=allow_union))
            return draw(st_ak.contents.record_array_contents(children))

        # Single-child wrapper
        child = _build(depth + 1, allow_union=allow_union)
        if node_type == 'regular':
            return draw(st_ak.contents.regular_array_contents(child))
        if node_type == 'list_offset':
            return draw(st_ak.contents.list_offset_array_contents(child))
        return draw(st_ak.contents.list_array_contents(child))

    return _build(0, allow_union=allow_union)
```

## Key Differences from Bottom-Up

| Aspect            | Bottom-up                          | Top-down                           |
| ----------------- | ---------------------------------- | ---------------------------------- |
| Node type chosen  | After children are built           | Before children are built          |
| Child count       | Emergent from "another edge?" loop | Coin flip per child after minimum  |
| No-nested-union   | `isinstance` check after the fact  | Structural via `allow_union=False` |
| Orphaned children | Possible; fallback discards them   | Impossible by construction         |
| Candidate logic   | `_candidate_node_types()` function | Inline from `allow_*` flags        |
| Child constraints | None; children built blindly       | Parent passes constraints down     |
| Budget exhaustion | Children built regardless          | Can stop adding children early     |

## What Is Eliminated

- `_candidate_node_types()` function
- Fallback/discard logic for orphaned children
- `isinstance(c, UnionArray)` runtime check
- `UnionArray` import (only needed for the runtime check)

## Open Questions

1. **`allow_union` parameter threading** — Only `union` restricts its children.
   If future node types add constraints, `_build` would need more parameters.
   A `frozenset` of allowed types could generalize this.
