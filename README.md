# EchoZero 2

Status: active
Last verified: 2026-04-30


EchoZero is now trimmed to the EZ2 app lane in this branch.

The canonical app surfaces in this repo are:
- `run_echozero.py` for the Stage Zero desktop shell
- `echozero/` for EZ2 core, application, UI, and Foundry code
- `python -m echozero.foundry.cli --root . ui` for the Foundry desktop UI

## Quick Start

```bash
python3 scripts/dev_bootstrap.py
```

EchoZero requires Python 3.11 or newer. On older macOS installs, `python3`
may still point to 3.9. If it does, install Python 3.11+ and create the venv
with that interpreter instead, for example `python3.11 -m venv .venv`.
On macOS, the quickest fix is usually `brew install python@3.11`.

If you want optional stacks during bootstrap:

```bash
python3 scripts/dev_bootstrap.py --extras ml packaging
```

Run the canonical EZ2 app shell:

```bash
.venv/bin/python run_echozero.py
```

Run Foundry:

```bash
pip install -e ".[ml]"
python -m echozero.foundry.cli --root . ui
```

Install the full local environment, including ML and packaging tools:

```bash
pip install -r requirements.txt
```

Run a small validation slice:

```bash
.venv/bin/python -m echozero.testing.run --lane appflow
```

Install local git hooks:

```bash
.venv/bin/pre-commit install
```

## Repo Map

```text
echozero/                EZ2 core application, Stage Zero UI, Foundry, shared runtime
tests/                   EZ2-heavy test suite
docs/                    Canonical plans, architecture context, cleanup map
run_echozero.py          Canonical EZ2 desktop launcher
deploy/                  Auth/models worker sidecars
```

## Cleanup Status

The current cleanup map is in [docs/EZ2-CODEBASE-CLEANUP-MAP.md](docs/EZ2-CODEBASE-CLEANUP-MAP.md).
Legacy EZ1 historical docs have been removed from this branch.
Generated tracking reports and local DB snapshots are no longer tracked.

The canonical implementation/architecture docs are:
- [docs/UNIFIED-IMPLEMENTATION-PLAN.md](docs/UNIFIED-IMPLEMENTATION-PLAN.md)
- [docs/APP-DELIVERY-PLAN.md](docs/APP-DELIVERY-PLAN.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [AGENTS.md](AGENTS.md)
- [docs/AGENT-CONTEXT.md](docs/AGENT-CONTEXT.md)
- [docs/agent-task-template.md](docs/agent-task-template.md)

## Important Note

Legacy EZ1 code paths such as `src/`, `ui/qt_gui/`, `main.py`, and `main_qt.py` have been removed from this branch. If a remaining doc still mentions them, treat that as historical drift rather than an active app surface.
