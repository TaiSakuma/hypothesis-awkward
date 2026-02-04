# Direct Constructors (Layouts) Research

**Date:** 2026-02-04
**Purpose:** Inform the design of an `arrays()` strategy that generates arbitrary
Awkward Arrays via direct Content constructors

## Motivation

Previous work explored two intermediate approaches:

1. **Types** (`.design/research/2026-01-21-type-system-research.md`) -- 8 Type
   classes describing what data looks like; lossy (many Forms per Type)
2. **Forms** (`.design/research/2026-02-03-form-system-research.md`) -- 12 Form
   classes describing structure; requires `ak.from_buffers()` to produce arrays

This note explores a third approach: building arrays directly from the 12
**Content** (layout) classes in `ak.contents`. These are the "Direct constructors
(fastest)" described in the Awkward Array documentation.

### Why Direct Constructors

- **One-step generation** -- produces real arrays with data, not intermediate
  descriptors
- **No roundtrip** -- avoids the Type-to-Form-to-buffer pipeline
- **Full coverage** -- exercises all 4 option representations, both list
  representations, etc.
- **Constructor validation** -- invalid nesting raises errors at construction time,
  so the strategy can lean on built-in checks
- **Trivial wrapping** -- `ak.Array(layout)` is zero-cost

## Content Class Catalog

Awkward Array v2.8.11 has 12 Content subclasses plus a base class:

```text
Content (base class)
+-- NumpyArray        - Primitive/multidimensional numeric data (leaf)
+-- EmptyArray        - Zero-length placeholder (leaf)
+-- RegularArray      - Fixed-size lists
+-- ListOffsetArray   - Variable-length lists (offsets buffer)
+-- ListArray         - Variable-length lists (starts + stops buffers)
+-- RecordArray       - Named-field records or unnamed tuples
+-- IndexedArray      - Re-indexed view (type-transparent)
+-- IndexedOptionArray - Nullable via index (-1 = missing)
+-- ByteMaskedArray   - Nullable via byte mask
+-- BitMaskedArray    - Nullable via bit mask
+-- UnmaskedArray     - Non-nullable option wrapper
+-- UnionArray        - Tagged union of multiple content types
```

**Location:** `awkward/contents/` directory, one file per class.

### Leaf Nodes (No Children)

#### EmptyArray

```python
EmptyArray(*, parameters=None, backend=None)
```

- Always length 0
- Cannot have parameters (raises `TypeError` if non-None/non-empty)
- Type: `UnknownType`
- No buffers

#### NumpyArray

```python
NumpyArray(data, *, parameters=None, backend=None)
```

- `data`: NumPy ndarray (1-d or multidimensional, must not be a scalar)
- Allowed dtypes: `bool`, `int8`/`16`/`32`/`64`, `uint8`/`16`/`32`/`64`,
  `float16`/`32`/`64`/`128`, `complex64`/`128`/`256`, `datetime64`,
  `timedelta64` (native endianness only; `float16`/`float128`/`complex256`
  availability depends on system)
- If `__array__` parameter is `"char"` or `"byte"`: data must be 1-d uint8
- Type: `NumpyType` (multidimensional data gets wrapped in `RegularType` layers)
- Buffers: `data`

### List Nodes (1 Child: `content`)

#### RegularArray

```python
RegularArray(content, size, zeros_length=0, *, parameters=None)
```

- `content`: Content subclass (flattened data)
- `size`: non-negative integer (fixed length of each list)
- `zeros_length`: non-negative integer; used as array length only when `size == 0`
- Length: `len(content) // size` when `size != 0`; `zeros_length` when `size == 0`
- No buffers (size is metadata)

#### ListOffsetArray

```python
ListOffsetArray(offsets, content, *, parameters=None)
```

- `offsets`: `Index` with dtype in `{int32, uint32, int64}`
- `content`: Content subclass
- Length: `len(offsets) - 1`
- Constraints: `len(offsets) >= 1`; `offsets[i] <= offsets[i+1]`; all values >= 0;
  values <= `len(content)`
- Buffers: `offsets`
- Most common list form

#### ListArray

```python
ListArray(starts, stops, content, *, parameters=None)
```

- `starts`: `Index` with dtype in `{int32, uint32, int64}`
- `stops`: `Index` with **same dtype** as starts
- `content`: Content subclass
- Length: `len(starts)`
- Constraints: `len(stops) >= len(starts)`; `starts[i] <= stops[i]`;
  `starts[i] >= 0`; `stops[i] <= len(content)`
- Buffers: `starts`, `stops`
- More general than `ListOffsetArray` -- content can be out of order, can have
  unreachable elements

### Record Node (0+ Children: `contents`)

#### RecordArray

```python
RecordArray(contents, fields, length=None, *, parameters=None, backend=None)
```

- `contents`: iterable of Content subclasses
- `fields`: list of strings or `None` (`None` = tuple, not record)
- `length`: explicit length; `None` means compute from shortest content
- Constraints: when `fields` is not None, `len(fields) == len(contents)`;
  when `length` is None and `len(contents) == 0`, raises `TypeError` (must
  specify length for empty records); when `length` is given, each content must
  have length >= `length`
- No buffers
- `__record__` parameter enables behavior dispatch

### Index Node (1 Child: `content`)

#### IndexedArray

```python
IndexedArray(index, content, *, parameters=None)
```

- `index`: `Index` with dtype in `{int32, uint32, int64}`
- `content`: Content subclass
- All index values must be `0 <= index[i] < len(content)` (no negative values)
- Length: `len(index)`
- Type: transparent -- returns `content.type` (no `IndexedType` exists)
- Buffers: `index`
- **Content restriction:** content cannot be union-type (unless
  `__array__` = `"categorical"`), option-type, or indexed-type

### Option Nodes (1 Child: `content`)

All option nodes share the same content restriction: content cannot be union-type
(unless categorical), option-type, or indexed-type.

#### IndexedOptionArray

```python
IndexedOptionArray(index, content, *, parameters=None)
```

- `index`: `Index` with dtype in **`{int32, int64}` only** (no uint32)
- `content`: Content subclass
- Negative index values indicate missing (None); non-negative are
  `0 <= index[i] < len(content)`
- Length: `len(index)`
- Type: `OptionType(content.type)`
- Buffers: `index`
- Default option representation from `from_type()`

#### ByteMaskedArray

```python
ByteMaskedArray(mask, content, valid_when, *, parameters=None)
```

- `mask`: `Index` with dtype = **int8 only**
- `content`: Content subclass
- `valid_when`: boolean -- if `True`, mask=1 means valid; if `False`, mask=1
  means missing
- Constraint: `len(mask) <= len(content)`
- Length: `len(mask)`
- Type: `OptionType(content.type)`
- Buffers: `mask`

#### BitMaskedArray

```python
BitMaskedArray(mask, content, valid_when, length, lsb_order, *, parameters=None)
```

- `mask`: `Index` with dtype = **uint8 only**
- `content`: Content subclass
- `valid_when`: boolean
- `length`: non-negative integer (needed because bits pack into groups of 8)
- `lsb_order`: boolean -- `True` = least-significant bit first
- Constraints: `length <= len(mask) * 8`; `length <= len(content)`
- Length: `length` parameter
- Type: `OptionType(content.type)`
- Buffers: `mask` (size = `ceil(length / 8)` bytes)
- Most compact option representation

#### UnmaskedArray

```python
UnmaskedArray(content, *, parameters=None)
```

- `content`: Content subclass
- Length: `len(content)`
- Type: `OptionType(content.type)` -- formally optional, but no actual missing
  values
- No buffers

### Union Node (2+ Children: `contents`)

#### UnionArray

```python
UnionArray(tags, index, contents, *, parameters=None)
```

- `tags`: `Index` with dtype = **int8 only**
- `index`: `Index` with dtype in `{int32, uint32, int64}`
- `contents`: list of Content subclasses (minimum 2)
- Constraints: `len(tags) <= len(index)` (usually equal);
  `0 <= tags[i] < len(contents)`;
  `0 <= index[i] < len(contents[tags[i]])`
- **Content restriction:** no union content (no nested unions); option content
  only if ALL contents are option; non-categorical indexed content is not allowed
- Maximum: 128 contents (because tags are int8, values 0-127)
- Length: `len(tags)`
- Type: `UnionType([c.type for c in contents])`
- Buffers: `tags`, `index`
- Formula: element at position `i` = `contents[tags[i]][index[i]]`

## Index Classes

All structure buffers are wrapped in `ak.index.Index` objects.

**Location:** `awkward/index.py`

| Class      | dtype  | Used by                                             |
| ---------- | ------ | --------------------------------------------------- |
| `Index8`   | int8   | `ByteMaskedArray.mask`, `UnionArray.tags`           |
| `IndexU8`  | uint8  | `BitMaskedArray.mask`                               |
| `Index32`  | int32  | offsets, starts, stops, index (32-bit accepting)    |
| `IndexU32` | uint32 | offsets, starts, stops, index (32-bit accepting)    |
| `Index64`  | int64  | offsets, starts, stops, index (default/most common) |

Index data must be one-dimensional and contiguous.

### Accepted dtypes per parameter

| Content Class        | Parameter | Accepted dtypes                   |
| -------------------- | --------- | --------------------------------- |
| `ListOffsetArray`    | offsets   | int32, uint32, int64              |
| `ListArray`          | starts    | int32, uint32, int64              |
| `ListArray`          | stops     | same dtype as starts              |
| `IndexedArray`       | index     | int32, uint32, int64              |
| `IndexedOptionArray` | index     | **int32, int64 only** (no uint32) |
| `ByteMaskedArray`    | mask      | **int8 only**                     |
| `BitMaskedArray`     | mask      | **uint8 only**                    |
| `UnionArray`         | tags      | **int8 only**                     |
| `UnionArray`         | index     | int32, uint32, int64              |

## Content Nesting Constraints

The direct constructors enforce strict nesting rules:

### Option/Indexed content restrictions

Applies to `IndexedArray`, `IndexedOptionArray`, `ByteMaskedArray`,
`BitMaskedArray`, `UnmaskedArray`. Their content **cannot** be:

- union-type (unless the wrapper has `__array__` = `"categorical"`)
- option-type
- indexed-type

If you need these combinations, use `ClassName.simplified()` which
flattens/restructures.

### Union content restrictions

Applies to `UnionArray`. Its contents **cannot** contain:

- union-type (no nested unions)
- non-categorical indexed-type
- option content is allowed only if **all** contents are option (all or none)

### No restrictions

`ListOffsetArray`, `ListArray`, `RegularArray`, `RecordArray`, `NumpyArray` can
wrap any Content.

### Summary table

| Parent         | Forbidden children                           |
| -------------- | -------------------------------------------- |
| Option nodes   | option, indexed, union                       |
| `IndexedArray` | option, indexed, union                       |
| `UnionArray`   | union, non-categorical indexed; mixed option |
| List nodes     | (none)                                       |
| `RecordArray`  | (none)                                       |

## Tree Structure

```text
Leaf nodes (no children):
  NumpyArray         -- actual data values
  EmptyArray         -- zero-length placeholder

Single-content nodes (1 child):
  RegularArray       -- fixed-size lists
  ListOffsetArray    -- variable-length lists (1 buffer)
  ListArray          -- variable-length lists (2 buffers)
  IndexedArray       -- re-indexing layer (type-transparent)
  IndexedOptionArray -- nullable via index
  ByteMaskedArray    -- nullable via byte mask
  BitMaskedArray     -- nullable via bit mask
  UnmaskedArray      -- formally nullable, but no actual Nones

Multi-content nodes (0+ or 2+ children):
  RecordArray        -- 0 or more fields (product type)
  UnionArray         -- 2 or more alternatives (sum type)
```

## Comparison with Types and Forms Approaches

| Aspect              | Types                            | Forms                    | Direct Constructors     |
| ------------------- | -------------------------------- | ------------------------ | ----------------------- |
| Abstraction         | What data is                     | What structure is        | Actual data + structure |
| Carries data?       | No                               | No                       | Yes                     |
| 1:1 with layout?    | No                               | Yes                      | They ARE layouts        |
| Class count         | 8                                | 12                       | 12                      |
| IndexedArray?       | N/A (transparent)                | IndexedForm exists       | IndexedArray exists     |
| Option variants     | 1 (OptionType)                   | 4 forms                  | 4 content classes       |
| List variants       | 2 types                          | 3 forms                  | 3 content classes       |
| Nesting constraints | None                             | None                     | Strict                  |
| Steps to array      | Type -> Form -> buffers -> array | Form -> buffers -> array | Layout -> array         |

The key trade-off: Types and Forms allow any composition (no nesting rules), but
direct constructors enforce strict constraints. This means a strategy must know the
rules -- but also means generated arrays are always valid.

## Buffer Generation Patterns

Each non-leaf node requires specific buffer invariants:

| Node                 | Buffer pattern                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ListOffsetArray`    | Sorted non-negative offsets of length `n+1` where `offsets[-1] <= len(content)`                             |
| `ListArray`          | `(start, stop)` pairs where `0 <= start <= stop <= len(content)`                                            |
| `RegularArray`       | No buffer; generate content of length `n * size`                                                            |
| `IndexedArray`       | Indices in `[0, len(content))`                                                                              |
| `IndexedOptionArray` | Indices in `[-1, len(content))` where -1 = missing                                                          |
| `ByteMaskedArray`    | int8 array of 0/1 values                                                                                    |
| `BitMaskedArray`     | uint8 array, `ceil(length / 8)` bytes                                                                       |
| `UnionArray`         | Consistent `(tags, index)` pairs where `tags[i]` selects a content and `index[i]` is valid for that content |
| `RecordArray`        | No buffer; multiple contents of compatible lengths                                                          |

## Strategy Design Implications

### Recursive generation pattern

A natural structure for the strategy:

- **Base case** (max depth reached): `NumpyArray` with random data, or `EmptyArray`
- **Recursive case:** choose from allowed node types, generate child content(s)
  recursively, then wrap with the chosen node type
- **Constraint enforcement:** when generating children for option/indexed nodes,
  exclude option/indexed/union content; when generating children for union nodes,
  exclude union content

### Content nesting rules for strategy

When generating a child content for different parent types, the strategy must
filter out certain child types:

| Parent type          | Forbidden child types                                                                |
| -------------------- | ------------------------------------------------------------------------------------ |
| `IndexedArray`       | `IndexedArray`, all 4 option nodes, `UnionArray`                                     |
| `IndexedOptionArray` | `IndexedArray`, all 4 option nodes, `UnionArray`                                     |
| `ByteMaskedArray`    | `IndexedArray`, all 4 option nodes, `UnionArray`                                     |
| `BitMaskedArray`     | `IndexedArray`, all 4 option nodes, `UnionArray`                                     |
| `UnmaskedArray`      | `IndexedArray`, all 4 option nodes, `UnionArray`                                     |
| `UnionArray`         | `UnionArray`, non-categorical `IndexedArray`; option only if ALL contents are option |
| All others           | No restrictions                                                                      |

This can be modeled by passing an `allowed` set or exclusion flags when recursing.

### Advantages

1. One-step generation produces actual arrays
2. Exercises more code paths than the canonical Forms from `from_type()`
3. Constructor validation catches invalid nesting
4. `ak.Array(layout)` wrapping is trivial

### Challenges

1. Index buffer generation must satisfy specific invariants (sorted offsets, valid
   ranges, etc.)
2. Length coordination -- content lengths must be consistent with wrapping structure
3. Nesting constraints must be tracked during recursive generation
4. Combinatorial explosion -- 12 classes with recursive composition

## Sources

- [How to create arrays -- Awkward Array documentation](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html)
- [Awkward Array source code](https://github.com/scikit-hep/awkward) v2.8.11
- `awkward/contents/*.py` -- all 12 Content subclass implementations
- `awkward/index.py` -- Index classes
