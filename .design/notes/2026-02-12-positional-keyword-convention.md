# Positional vs Keyword-Only Parameter Convention

**Date:** 2026-02-12
**Status:** Proposed

## Hypothesis Pattern

Hypothesis strategies consistently separate **positional** parameters (the
"what") from **keyword-only** configuration (the "how") using `*`:

```python
# "What" parameter is positional; config is keyword-only
def lists(elements, *, min_size=0, max_size=None, unique_by=None, unique=False)
def text(alphabet=characters(), *, min_size=0, max_size=None)
def from_regex(regex, *, fullmatch=False)
def dictionaries(keys, values, *, dict_class=dict, min_size=0, max_size=None)

# numpy extras follow the same rule
def arrays(dtype, shape, *, elements=None, fill=None, unique=False)
def from_dtype(dtype, *, allow_nan=None, allow_infinity=None, ...)

# Bounds can be positional-capable when they ARE the strategy's purpose
def integers(min_value=None, max_value=None)
def floats(min_value=None, max_value=None, *, allow_nan=None, ...)

# All-config strategies use keyword-only throughout
def binary(*, min_size=0, max_size=None)
def complex_numbers(*, min_magnitude=0, max_magnitude=None, allow_nan=None, ...)
```

### Rules

1. **"What" parameter** (the primary input the strategy operates on) is
   positional or positional-capable — placed before `*`.
   - Required: `lists(elements)`, `from_dtype(dtype)`, `arrays(dtype, shape)`
   - Optional with default: `text(alphabet=characters())`

2. **Configuration** (flags, bounds, limits) is keyword-only — placed after `*`.
   - Boolean flags: `allow_nan`, `unique`, `fullmatch`
   - Size bounds: `min_size`, `max_size`
   - Other config: `width`, `dict_class`

3. **Exception**: When bounds _are_ the purpose (e.g., `integers(min_value,
   max_value)`), they can be positional-capable with no `*`.

4. **Positional-only (`/`)**: Only used for very special cases
   (`builds(target, /)`).

## Current State of hypothesis_awkward Strategies

### Already follows the convention

| Strategy        | Signature               | Notes                          |
| --------------- | ----------------------- | ------------------------------ |
| `none_or`       | `(st_)`                 | Single positional param        |
| `ranges`        | `(st_, /, *, ...)`      | Positional-only + keyword-only |
| `leaf_contents` | `(*, dtypes=None, ...)` | All keyword-only               |

### Needs `*` separator added

#### Group A: Has a "what" parameter (positional + keyword-only config)

These have a primary input that should stay positional, with config after `*`:

| Strategy                     | Current                          | Proposed                            |
| ---------------------------- | -------------------------------- | ----------------------------------- |
| `items_from_dtype`           | `(dtype, allow_nan=False)`       | `(dtype, *, allow_nan=False)`       |
| `numpy_forms`                | `(type_=None, dtypes=None, ...)` | `(type_=None, *, dtypes=None, ...)` |
| `regular_array_contents`     | `(draw, content=None)`           | `(draw, content=None, *)`           |
| `list_offset_array_contents` | `(draw, content=None)`           | `(draw, content=None, *)`           |
| `list_array_contents`        | `(draw, content=None)`           | `(draw, content=None, *)`           |

**Rationale:**

- `items_from_dtype(dtype)` — `dtype` is required, like `from_dtype(dtype, *,
  ...)`
- `numpy_forms(type_=None)` — `type_` is the primary constraint ("generate a
  form from this type"). Optional with default, like `text(alphabet=..., *,
  ...)`. When given, it determines the entire output and other params are
  ignored.
- `regular_array_contents(content=None)` — `content` is what you're wrapping,
  like `lists(elements, *, ...)`. Optional because `None` means "generate
  content internally". No keyword-only params exist yet, but the `*` future-
  proofs the signature.

#### Group B: All configuration (keyword-only throughout)

These have no clear "what" parameter — everything is optional configuration:

| Strategy               | Current                              | Proposed                                |
| ---------------------- | ------------------------------------ | --------------------------------------- |
| `numpy_types`          | `(dtypes=None, allow_datetime=True)` | `(*, dtypes=None, allow_datetime=True)` |
| `numpy_arrays`         | `(draw, dtype=None, ...)`            | `(draw, *, dtype=None, ...)`            |
| `from_numpy`           | `(dtype=None, ...)`                  | `(*, dtype=None, ...)`                  |
| `numpy_dtypes`         | `(dtype=None, ...)`                  | `(*, dtype=None, ...)`                  |
| `numpy_array_contents` | `(dtypes=None, ...)`                 | `(*, dtypes=None, ...)`                 |
| `contents`             | `(draw, dtypes=None, ...)`           | `(draw, *, dtypes=None, ...)`           |
| `arrays`               | `(draw, dtypes=None, ...)`           | `(draw, *, dtypes=None, ...)`           |
| `lists`                | `(draw, dtype=None, ...)`            | `(draw, *, dtype=None, ...)`            |
| `from_list`            | `(dtype=None, ...)`                  | `(*, dtype=None, ...)`                  |
| `dicts_for_dataframe`  | `(draw, max_columns=4, ...)`         | `(draw, *, max_columns=4, ...)`         |

**Rationale:**

- `dtypes`/`dtype` in these strategies is an optional constraint, not a primary
  input. It defaults to `None` (meaning "use supported_dtypes()"). Calling
  `numpy_arrays(some_dtype)` positionally would be valid but these strategies
  have many optional params where positional calling would be error-prone.
- Compare with `st.complex_numbers(*, ...)` and `st.binary(*, ...)` in
  Hypothesis — all-config strategies use keyword-only throughout.

### No changes needed

| Strategy                   | Signature | Notes     |
| -------------------------- | --------- | --------- |
| `supported_dtypes`         | `()`      | No params |
| `supported_dtype_names`    | `()`      | No params |
| `builtin_safe_dtypes`      | `()`      | No params |
| `builtin_safe_dtype_names` | `()`      | No params |
| `empty_array_contents`     | `()`      | No params |

## Edge Cases

### `numpy_forms(type_=None)` — Why positional?

`type_` is the primary constraint that fully determines the output. It parallels
`from_dtype(dtype)` in Hypothesis — "generate X from this constraint". The key
difference is that our `type_` is optional: when absent, other params take over.
This is analogous to `text(alphabet=characters())` where `alphabet` has a
default but is still the primary input.

Placing `type_` before `*` allows the natural call `numpy_forms(some_type)`.

### `dtype` vs `dtypes` — Why keyword-only?

In `numpy_arrays(dtype=None)`, `dtype` filters which dtypes are used. It's
optional and one of many configuration knobs. Calling
`numpy_arrays(np.dtype('f8'))` positionally works today, but as strategies gain
more parameters, positional calls become fragile. Making `dtype`/`dtypes`
keyword-only matches the "all-config" pattern.

Compare: `st.binary(*, min_size=0, max_size=None)` — even simple bounds are
keyword-only when there's no primary input.

### `@st.composite` and `draw`

For composite strategies, `draw` is always the first parameter and is stripped
from the public signature by Hypothesis. The `*` appears after the last
positional-capable user-facing parameter:

```python
@st.composite
def contents(
    draw: st.DrawFn,
    *,
    dtypes: ... | None = None,
    max_size: int = 10,
) -> Content:
```

The user sees: `contents(*, dtypes=None, max_size=10)`.

### Content strategies with `content` parameter

`regular_array_contents(content=None)` has no keyword-only params today. Adding
`*` after `content` has no immediate effect but future-proofs the signature:

```python
@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: ... | None = None,
    *,
) -> Content:
```

When future params are added (e.g., `max_size`), they go after `*` without
breaking existing callers.

## Summary

The convention is: **"what" is positional, "how" is keyword-only.** When
everything is "how", use keyword-only throughout.
