# UX Research Interview: hypothesis-awkward Developer

**Date:** 2026-01-21
**Participant:** The developer of hypothesis-awkward
**Interviewer:** Claude

## Project Vision

### Current State

- Developed experimental strategies: `from_numpy`, `from_list`, etc.
- Strategies are tested with Hypothesis (meta-testing pattern documented in `.claude/rules/testing-patterns.md`)

### Ultimate Goal

Develop an `arrays()` strategy that:

- Generates fully general Awkward Arrays by default
- Has many options to control the results

## Key Insights

### 1. API Design Inspiration

Inspired by `hypothesis.extra.numpy.arrays()` which has `dtype` and `shape` options that accept either:

- Specific values
- Strategies that generate values of that type

**Planned `arrays()` options:**

- `type` - Awkward Array type
- `form` - Awkward Array form
- Additional options: min/max values, numbers, etc.

**Challenge:** `type` and `form` are not independent. One type corresponds to multiple forms. Typically only one will be given by the user.

**Supporting strategies:**

- `types()` - generates types with many options
- `forms()` - generates forms with many options

### 2. Development Approach

- Iterative development, not one-shot
- **Next step:** Start with `types()` strategy
- Previous path: building-block strategies (`from_numpy`, `from_list`, potentially `from_pyarrow`)

### 3. Target Users

1. **Awkward Array developers** - for testing Awkward Array internals
2. **Developers of tools using Awkward Array** - e.g., scikit-HEP packages
3. **Physicists** - need precise control over array structure to match their analysis code

### 4. Success Criteria

- Used to find edge cases in tools that use Awkward Array and in Awkward Array itself

### 5. Edge Case Discovery

Property-based testing already proved valuable:

- Found an issue in `to_numpy()` for structured arrays ([awkward#3690](https://github.com/scikit-hep/awkward/issues/3690))
- Round-trip `np.array` → `ak.from_numpy` → `to_numpy` failed for 2D boolean structured arrays

## Open Questions / Challenges

### Type Enumeration Problem

Types can have many different nested structures, and understanding how to systematically enumerate and construct them is important.

This is a key technical challenge for implementing `types()`:

- How to systematically enumerate all valid type structures?
- How to construct them programmatically?

### Generating 'Interesting' Arrays

Two main challenges identified:

1. **Coverage:** Awkward Array has a large number of possible data structures
2. **API Design:** Designing the options for `arrays()` to give users appropriate control

## Action Items

- [ ] Research Awkward Array type system enumeration
- [ ] Design `types()` strategy API
- [ ] Consider how `types()` and `forms()` interact with `arrays()`
- [ ] Document API design decisions in `.design/` directory

## Summary

After building foundational strategies incrementally, there is now sufficient experience to begin designing the main `arrays()` strategy. Physicists in particular will need fine-grained control over array structure to match their specific analysis requirements.
