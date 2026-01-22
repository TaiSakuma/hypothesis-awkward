# Awkward Array Type System Research

**Date:** 2026-01-21
**Purpose:** Inform the design of `types()` strategy for hypothesis-awkward

## Type Class Hierarchy

Awkward Array has 10 type classes in `ak.types`:

```
Type (base class)
├── ArrayType      - Outermost wrapper with length (e.g., "5 * int64")
├── ScalarType     - Single element extracted from array
├── NumpyType      - Primitive types (int64, float32, bool, datetime64, etc.)
├── ListType       - Variable-length lists ("var * ...")
├── RegularType    - Fixed-length lists ("3 * ...")
├── RecordType     - Named fields {...} or tuples (...)
├── OptionType     - Nullable values ("?..." or "option[...]")
├── UnionType      - Multiple possible types ("union[A, B]")
└── UnknownType    - Placeholder for undetermined types
```

## Type Construction

### Programmatic Construction

Each type class has a constructor:

```python
import awkward as ak

# Primitives
t_int = ak.types.NumpyType('int64')
t_float = ak.types.NumpyType('float32')
t_dt = ak.types.NumpyType('datetime64[ns]')

# Lists
t_var_list = ak.types.ListType(t_int)                    # var * int64
t_reg_list = ak.types.RegularType(t_int, size=3)         # 3 * int64

# Records
t_record = ak.types.RecordType(
    contents=[t_int, t_float],
    fields=['x', 'y']
)  # {x: int64, y: float32}

t_tuple = ak.types.RecordType(
    contents=[t_int, t_float],
    fields=None  # None makes it a tuple
)  # (int64, float32)

# Options
t_optional = ak.types.OptionType(t_int)  # ?int64

# Unions
t_union = ak.types.UnionType([t_int, t_float])  # union[int64, float32]

# Array wrapper
t_array = ak.types.ArrayType(t_var_list, length=10)  # 10 * var * int64
```

### From Datashape Strings

```python
# Parse datashape notation
t = ak.types.from_datashape('var * {x: float64, y: ?int32}')
```

## NumpyType Primitives

Supported primitive strings:

- **Integers:** int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **Floats:** float16 (if available), float32, float64, float128 (if available)
- **Complex:** complex64, complex128, complex256 (if available)
- **Other:** bool, datetime64, timedelta64

Datetime/timedelta support units: `datetime64[ns]`, `timedelta64[15us]`, etc.

## Type Nesting Patterns

Types compose recursively. Common patterns in HEP:

```
# Simple array of integers
int64

# Jagged array (variable-length lists)
var * int64

# Doubly jagged
var * var * int64

# Record with fields
{pt: float64, eta: float64, phi: float64, mass: float64}

# Jagged array of records (typical HEP event data)
var * {pt: float64, eta: float64, phi: float64, charge: int32}

# Optional values
?float64

# Record with optional fields
{x: float64, y: ?float64}

# Union types (heterogeneous data)
union[{x: int64}, {y: int64}]

# Complex nested structure
var * {
    muons: var * {pt: float64, eta: float64},
    jets: var * {pt: float64, btag: bool}
}
```

## Type vs Form Relationship

From documentation:

> "There is a one-to-one relationship between `ak.contents.Content` subclasses and `ak.forms.Form` subclasses, and each `ak.forms.Form` maps to only one `ak.types.Type`."

**Key insight for strategy design:**

- One **Form** → One **Type** (deterministic)
- One **Type** → Multiple **Forms** (many implementations possible)

Example: `ListType` can be implemented by:

- `ListOffsetArray` (single offsets array)
- `ListArray` (starts + stops arrays)

## Real-World Usage Patterns

### HEP Physics Data (from GitHub discussions)

Typical muon array type:

```
var * {pt: float64, eta: float64, phi: float64, mass: float64}
```

With vector package registration:

```
var * Momentum4D[pt: float64, eta: float64, phi: float64, mass: float64]
```

### Type Checking in coffea

coffea uses `isinstance` checks with type classes:

```python
isinstance(aktype, ak.types.ArrayType)
isinstance(aktype.type, ak.types.ListType)
isinstance(aktype.type, ak.types.NumpyType)
```

### Type Enforcement

```python
# Convert datashape string to type, then enforce
type_obj = ak.types.from_datashape('var * {x: float64, y: float32}')
result = ak.enforce_type(my_array, type_obj)
```

## Enumeration Strategy for `types()`

### Recursive Structure

Types form a tree. Strategy should generate:

1. **Leaf types:** `NumpyType`, `UnknownType`
2. **Container types:** Wrap other types
   - `ListType(content)`
   - `RegularType(content, size)`
   - `OptionType(content)`
3. **Structured types:**
   - `RecordType(contents, fields)` - multiple children
   - `UnionType(contents)` - multiple alternatives
4. **Wrappers:** `ArrayType`, `ScalarType`

### Proposed Strategy Structure

```python
@st.composite
def types(
    draw,
    # Control which type classes to include
    allow_list: bool = True,
    allow_regular: bool = True,
    allow_record: bool = True,
    allow_option: bool = True,
    allow_union: bool = True,
    allow_unknown: bool = False,

    # Control nesting
    max_depth: int = 3,

    # Control primitives
    numpy_dtypes: st.SearchStrategy[np.dtype] | None = None,

    # Control records
    max_fields: int = 5,
    allow_tuple: bool = True,

    # Control lists
    max_regular_size: int = 10,
) -> ak.types.Type:
    ...
```

### Recursive Generation Pattern

```python
def types(max_depth: int = 3, ...):
    if max_depth <= 0:
        # Base case: only leaf types
        return numpy_types(...)

    # Recursive case: choose from all allowed types
    strategies = [numpy_types(...)]

    if allow_list:
        strategies.append(
            types(max_depth - 1, ...).map(ak.types.ListType)
        )

    if allow_record:
        strategies.append(record_types(max_depth - 1, ...))

    # etc.

    return st.one_of(*strategies)
```

## Special Considerations

### String Types

Strings are `ListType` with special parameters:

```python
# String: list of uint8 with __array__='char' parameter
ak.types.ListType(
    ak.types.NumpyType('uint8', parameters={'__array__': 'char'}),
    parameters={'__array__': 'string'}
)
```

Consider whether `types()` should have an `allow_string: bool` option.

### Categorical Types

Some types have `__categorical__` parameter for categorical data.

### Behavior Registration

Types can have custom behaviors (e.g., `Momentum4D` for physics vectors).
The `parameters` dict stores `__record__` for named record types.

## Sources

- [Awkward Array Documentation - How to examine type](https://awkward-array.org/doc/main/user-guide/how-to-examine-type.html)
- [Awkward Array Documentation - How to create lists](https://awkward-array.org/doc/main/user-guide/how-to-create-lists.html)
- [GitHub: scikit-hep/awkward source code](https://github.com/scikit-hep/awkward)
- [GitHub Discussion: Precision of awkward array #3218](https://github.com/scikit-hep/awkward/discussions/3218)
- [GitHub Discussion: Building arrays of specified dtype #328](https://github.com/scikit-hep/awkward/discussions/328)
- [GitHub: uproot5 discussions on type representations](https://github.com/scikit-hep/uproot5/discussions/903)
- [GitHub: vector package discussions](https://github.com/scikit-hep/vector/discussions/117)
