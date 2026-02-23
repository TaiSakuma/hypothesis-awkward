# `max_length` Research

Date: 2026-02-23

## Overview

This document explores adding a `max_length` parameter to `contents()` that
would cap the immediate `len()` of generated content, as opposed to the existing
`max_size` which caps the total scalar count across all leaves.

## Motivation

`regular_array_contents()` currently draws child content first, then finds valid
`size` divisors of `len(content)`. This biases `size` toward small values:

- Prime-length content (e.g., length 7) only allows `size=1`
- Large content lengths have few divisors relative to their magnitude
- Small `size` values like 1 are overrepresented because 1 divides every integer

Drawing `size` first would give more uniform coverage, but requires the ability
to control `len(content)` so that it is a multiple of the chosen `size`. This
means we need a way to constrain the immediate length of content independently
of the total scalar budget.

## Two Length Concepts

The codebase currently uses `max_size` to mean "total scalar count across all
leaf nodes in the content tree." This is distinct from the immediate `len()` of
a content node:

- **`max_size`** (total scalars): Counts every scalar value in every
  `NumpyArray` leaf, every string in `string_contents`, every bytestring in
  `bytestring_contents`. A `RegularArray` of size 3 wrapping a `NumpyArray` of
  length 12 has `max_size` cost of 12 (the leaf scalars), but `len()` of 4 (the
  number of groups).

- **`max_length`** (immediate length): The `len()` of a content node at a
  single level. For a `NumpyArray`, this equals the number of elements. For a
  `RegularArray`, this is `len(content) // size`. For a `ListOffsetArray`, this
  is `len(offsets) - 1`.

These are independent dimensions: a `RegularArray` with size 1 and content
length 100 has `max_length=100` and `max_size=100`, while size 10 and content
length 100 has `max_length=10` and `max_size=100`.

## Survey of Strategies

Which strategies have `min_size`/`max_size` and what they control:

| Strategy                       | `min_size` | `max_size`          | Controls                              |
| ------------------------------ | ---------- | ------------------- | ------------------------------------- |
| `contents()`                   | no         | yes                 | total scalars across all leaves       |
| `leaf_contents()`              | yes        | yes                 | immediate length (leaf is flat)       |
| `numpy_array_contents()`       | yes        | yes                 | immediate length (leaf is flat)       |
| `string_contents()`            | yes        | yes                 | number of strings                     |
| `bytestring_contents()`        | yes        | yes                 | number of bytestrings                 |
| `empty_array_contents()`       | no         | no                  | always 0                              |
| `regular_array_contents()`     | no         | yes (`max_size`)    | element size (group size, not length) |
| `list_offset_array_contents()` | no         | no                  | hardcoded `MAX_LIST_LENGTH=5`         |
| `list_array_contents()`        | no         | no                  | hardcoded `MAX_LIST_LENGTH=5`         |
| `record_array_contents()`      | no         | no (`max_fields`)   | number of fields                      |
| `union_array_contents()`       | no         | no (`max_contents`) | number of alternatives                |

Key finding: no strategy currently supports constraining the immediate `len()`
of its output independently of the total scalar budget. For leaf strategies,
`max_size` happens to equal immediate length (since leaves are flat). But for
wrapper strategies, there is no way to say "produce content with `len()` at most
N."

## Where `max_length` Would Need to Be Added

To be useful, `max_length` would need to be:

1. **Added to `contents()`** as a new parameter, separate from `max_size`.
   `contents()` would propagate it to whichever wrapper strategy it selects.

2. **Propagated to wrapper strategies**: Each wrapper strategy would need to
   accept `max_length` and constrain its output accordingly:
   - `regular_array_contents()` — constrain `len(content) // size`
   - `list_offset_array_contents()` — constrain number of sublists
   - `list_array_contents()` — constrain number of sublists
   - `record_array_contents()` — constrain `length` (all fields share it)
   - `union_array_contents()` — constrain `sum(len(c) for c in contents)`

3. **Interaction with `max_size`**: Both constraints must hold simultaneously.
   A content node must have `len() <= max_length` _and_ total leaf scalars
   `<= max_size`. This means the strategies need to coordinate: drawing content
   that satisfies the scalar budget while also hitting a specific length.

## Use Case: Improving `regular_array_contents()`

The primary motivation is to improve `size` coverage in
`regular_array_contents()`. With `max_length` on the child content, the strategy
could:

1. Draw `size` first from `[0, max_size]`
2. Draw a target length that is a multiple of `size`
3. Draw child content with `max_length=target_length`

This would give uniform coverage over `size` values instead of being biased by
the divisor structure of whatever content length happens to be drawn.

However, the recent refactoring of `regular_array_contents()` (extracting
`_st_group_sizes()`, adding `max_size` and `max_zeros_length` parameters)
already improves the situation somewhat by giving callers more control over
the generated arrays.

## Status

Research only. No implementation planned at this time. The complexity of
threading `max_length` through the entire contents hierarchy and coordinating it
with `max_size` may not be justified by the improvement in `size` coverage for
`regular_array_contents()`.
