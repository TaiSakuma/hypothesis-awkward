---
name: update-docs
description: Review and update project documentation
disable-model-invocation: true
---

Review the documentation in `.claude/` and `.design/` against the current state
of the repository. Update any docs that are outdated.

## Steps

1. **Explore the repo** — Use Glob and Read to understand the current source
   layout (`src/`), tests (`tests/`), and config files (`pyproject.toml`,
   `.github/`).

2. **Audit `.claude/`** — Read every file under `.claude/` (except `settings*`).
   For each file, check whether its content still accurately reflects the repo:
   - `CLAUDE.md` — project overview, build commands, code architecture,
     strategy listing, usage conventions, code style rules.
   - `rules/ci.md` — CI matrix, dependency files, release workflows.
   - `rules/testing-patterns.md` — test patterns, example code.
   - `rules/git.md` and `rules/markdown.md` — short rules.

3. **Audit `.design/`** — Read every file under `.design/`. For each file,
   check whether the described API, implementation plan, or research is still
   consistent with the current code. Flag anything that has diverged.

4. **Report findings** — Summarize what is outdated and what changes you
   propose. Group by file.

5. **Apply updates** — Edit the outdated docs to match reality. Keep the
   existing voice and style. Do not rewrite content that is still accurate.
   For `.design/` docs, add a note at the top if the design has been
   superseded or significantly changed, rather than rewriting history.

6. **Optimize context** — Review always-loaded context (files without
   `paths:` frontmatter) for size and relevance:
   - Check total line counts. Flag any rule file over ~50 lines that isn't
     path-scoped.
   - Check whether any section in `CLAUDE.md` contains details that Claude
     could discover from code (e.g., per-function listings). Suggest trimming.
   - Check whether any rule file should be path-scoped to reduce default
     context load.

7. **Format** — Run `dprint fmt` on any markdown files you changed.

## Guidelines

- Do not touch `settings.json` or `settings.local.json`.
- Prefer minimal, targeted edits over wholesale rewrites.
- If a `.design/` doc is fully implemented and accurate, leave it alone.
- If a `.design/` doc describes something that was implemented differently,
  add a brief "Status" note at the top explaining the divergence.
