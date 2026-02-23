---
paths:
  - ".github/**"
  - "pyproject.toml"
---

# CI and Dependency Management

## Dependency Pinning

The project uses two requirements files under `.github/deps/` for CI testing:

- **`minimum/requirements.txt`** — Pins the oldest supported versions (matching
  the lower bounds in `pyproject.toml`). Manually maintained.
- **`latest/requirements.txt`** — Pins the latest known versions. Monitored by
  Dependabot, which opens PRs to bump these pins.

These files are **not** used for local development. The project does not use
`uv.lock`.

## Updating Dependencies

- When bumping a lower bound in `pyproject.toml` (e.g., `awkward>=2.9`), also
  update `minimum/requirements.txt` to match.
- `latest/requirements.txt` is updated automatically by Dependabot.

## CI Matrix (`unit-test.yml`)

Runs on push to `main` and PRs targeting `main`. Tests a matrix of:

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **Deps**: `default`, `latest`, `min`
- **Optional deps**: none or `[all]` (pandas + pyarrow)

The `[all]` extra only runs with `default` deps (excluded from `latest`/`min`).

| Mode      | Install method                                                    |
| --------- | ----------------------------------------------------------------- |
| `default` | `uv pip install -e .[opt-deps] --group dev` (free resolution)     |
| `latest`  | Installs package, then overwrites with `latest/requirements.txt`  |
| `min`     | Installs package, then downgrades with `minimum/requirements.txt` |

## Release Workflows

Two workflows triggered by pushing a `v*.*.*` tag:

- **`pypi.yml`** — Builds with `hatch build`, publishes to PyPI via trusted
  publishing.
- **`release.yml`** — Creates a GitHub Release with auto-generated notes and
  updates a `latest` tag.
