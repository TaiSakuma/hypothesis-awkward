# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

hypothesis-awkward is a Python library providing
[Hypothesis](https://hypothesis.works/) strategies for generating [Awkward Array](https://awkward-array.org/) test data. It enables property-based testing
for code that uses Awkward Arrays.

## Build and Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[Hatch](https://hatch.pypa.io/) for building.

```bash
# Install in development mode with all optional dependencies
uv pip install -e .[all] --group dev

# Run all tests with coverage
uv run pytest -vv --cov

# Run a single test file
uv run pytest tests/strategies/numpy/test_from_numpy.py -vv

# Run a specific test
uv run pytest tests/strategies/numpy/test_from_numpy.py::test_from_numpy -vv

# Type checking
uv run mypy src

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests
```

## Code Architecture

### Source Layout (`src/hypothesis_awkward/`)

- **`strategies/`** - Public Hypothesis strategies (imported as
  `hypothesis_awkward.strategies`)
  - `numpy/` - Strategies for Awkward Arrays from NumPy: `from_numpy`,
    `numpy_arrays`, `numpy_dtypes`, `supported_dtype_names`, `supported_dtypes`
  - `builtins_/` - Strategies for Awkward Arrays from Python lists: `from_list`,
    `lists`, `items_from_dtype`, `builtin_safe_dtypes`
  - `contents/` - Strategies for Awkward Array content layouts:
    `contents`, `leaf_contents`, `numpy_array_contents`,
    `empty_array_contents`, `regular_array_contents`,
    `list_offset_array_contents`, `list_array_contents`
  - `constructors/` - Strategies via direct Content constructors: `arrays`
  - `pandas/` - Strategies related to pandas DataFrames: `dicts_for_dataframe`
  - `forms/` - (experimental) Strategies for Awkward Forms: `numpy_forms`
  - `types/` - (experimental) Strategies for Awkward Types: `numpy_types`
  - `misc/` - Utility strategies: `ranges`, `none_or`, `StMinMaxValuesFactory`,
    `Opts`, `RecordDraws`

- **`util/`** - Internal utilities for dtype handling, safe comparisons, and
  array introspection

### Strategy Design Pattern

Each strategy module follows a consistent pattern:

1. A "raw" strategy generating intermediate data (e.g., `numpy_arrays` generates
   NumPy arrays)
2. A convenience strategy that converts to Awkward Array (e.g., `from_numpy`
   wraps `numpy_arrays` and calls `ak.from_numpy`)

### Usage Convention

Strategies are imported as:

```python
import hypothesis_awkward.strategies as st_ak
```

## Testing

Tests mirror the source structure under `tests/`. The project uses pytest with
doctest support for examples in docstrings and markdown files.

## Code Style

- Use single quotes (`'''`) for docstrings.
- Prefer single quotes (`'`) for strings.
- Use absolute imports for parent packages (e.g., `from hypothesis_awkward.strategies.numpy import numpy_arrays`, not `from ..numpy import numpy_arrays`). Same-package relative imports (e.g., `from .arrays_ import arrays`) are fine.
- For forward references in type annotations, prefer string quotes (e.g., `-> 'MyClass[Any]'`) over `from __future__ import annotations`.
- Sort imports with `uv run ruff check --select I --fix src tests`.
- Run `uv run mypy src tests` after making changes to verify type correctness.
