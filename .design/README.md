# Design Documents

This directory contains UX research findings and API design documents for
hypothesis-awkward.

## Directory Structure

```text
.design/
├── README.md          # This file
├── research/          # UX research interviews and findings
├── api/               # API design proposals and decisions
├── impl/              # Implementation decisions (directory structure, internals)
└── notes/             # Progress tracking and working notes
```

## Purpose

- **research/**: Store UX research interview notes and insights. These inform API
  design decisions by capturing user needs, pain points, and use cases.

- **api/**: Store API design proposals, alternatives considered, and final
  decisions with rationale.

- **impl/**: Store implementation decisions that don't affect the user-facing API,
  such as directory structure, module organization, and internal architecture.

- **notes/**: Store progress tracking and working notes for ongoing work.

## Naming Convention

- Research files: `YYYY-MM-DD-ux-interview-NN.md`
- API design files: `YYYY-MM-DD-<feature>-api.md`
- Implementation files: `YYYY-MM-DD-<topic>.md`
