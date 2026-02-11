# Types Strategy Tests Progress

**Date:** 2026-01-22
**Updated:** 2026-02-09
**Status:** Paused (focus shifted to constructors/arrays)

## Completed

- [tests/strategies/types/test_numpy_types.py](../../tests/strategies/types/test_numpy_types.py) - Done
- `numpy_types()` strategy implemented and tests pass

## Resolved: Testing `content` Strategy Parameter

The open question about testing strategy-valued parameters (like
`list_types(content=...)`) was resolved in later work. The `st_ak.RecordDraws`
pattern was adopted:

- Wrap the strategy in `st_ak.RecordDraws(...)` to record drawn values
- Use `st_ak.Opts` with `reset()` to clear recorders between draws
- Use `match` / `case` in assertions to distinguish concrete values from
  `RecordDraws` instances

This pattern is now documented in
[testing-patterns.md](./../../.claude/rules/testing-patterns.md) and used in:

- `tests/strategies/forms/test_numpy_forms.py`
- `tests/strategies/constructors/test_array.py`

## Remaining Tests to Write

Per the API design doc order:

1. ~~`numpy_types()`~~ ✓
2. `list_types()` (draft was removed in commit 7681f31)
3. `regular_types()`
4. `option_types()`
5. `record_types()`
6. `union_types()`
7. `string_types()` / `bytestring_types()`
8. `types()` (main recursive strategy)

## Notes

- The `list_types()` draft test file was removed (commit 7681f31) along with
  `pragma: no cover` cleanup. The corresponding `list_types()` strategy was
  never implemented.
- Development focus shifted to the `constructors/arrays()` strategy, which
  generates actual arrays via direct Content constructors rather than going
  through the Type -> Form -> buffer pipeline.
