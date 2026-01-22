# Types Strategy Tests Progress

**Date:** 2026-01-22
**Status:** In progress

## Completed

- [tests/strategies/types/test_numpy_types.py](../../tests/strategies/types/test_numpy_types.py) - Done and staged
- `numpy_types()` strategy implemented and tests pass

## In Progress

- [tests/strategies/types/test_list_types.py](../../tests/strategies/types/test_list_types.py) - Draft, not staged

### Open Question: Testing `content` parameter

The `list_types(content=...)` parameter accepts a `SearchStrategy[ak.types.Type]`.
This creates a testing challenge:

- We can't verify that the result came from a specific strategy
- Unlike boolean flags (`allow_nan=False` → no NaN in result), a strategy
  parameter doesn't have an easily testable constraint

**Current approach:**

- `list_types_kwargs()` generates `content` as either `None` or
  `st.just(st_ak.numpy_types())`
- Main test only verifies: if `content=None`, result.content is `NumpyType`
- `find()` tests cover custom content case

**Possible alternatives:**

1. Remove `content` from kwargs strategy entirely (only test default)
2. Generate a concrete `Type` and wrap it in `st.just()` for testing
3. Accept that strategy parameters aren't fully testable in main property test

This same issue will apply to: `regular_types()`, `option_types()`,
`record_types()`, `union_types()` - all have `content` strategy parameters.

## Remaining Tests to Write

Per the API design doc order:

1. ~~`numpy_types()`~~ ✓
2. `list_types()` ← current (draft exists)
3. `regular_types()`
4. `option_types()`
5. `record_types()`
6. `union_types()`
7. `string_types()` / `bytestring_types()`
8. `types()` (main recursive strategy)

## Next Session Prompt

```text
Continue tests for types strategies.

See .design/notes/2026-01-22-types-tests-progress.md for status.

Open question: how to test `content` strategy parameter in list_types().
Current draft: tests/strategies/types/test_list_types.py (not staged)
```
