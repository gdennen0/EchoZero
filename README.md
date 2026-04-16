# EchoZero 2

EchoZero is now trimmed to the EZ2 app lane in this branch.

The canonical app surfaces in this repo are:
- `run_echozero.py` for the Stage Zero desktop shell
- `echozero/` for EZ2 core, application, UI, and Foundry code
- `python -m echozero.foundry.cli --root . ui` for the Foundry desktop UI

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the canonical EZ2 app shell:

```bash
python run_echozero.py
```

Run Foundry:

```bash
python -m echozero.foundry.cli --root . ui
```

Run a small validation slice:

```bash
pytest tests/testing/test_app_shell_profiles.py tests/ui/test_run_echozero_launcher.py -q
```

## Repo Map

```text
echozero/                EZ2 core application, Stage Zero UI, Foundry, shared runtime
tests/                   EZ2-heavy test suite
docs/                    Canonical plans, architecture context, cleanup map
run_echozero.py          Canonical EZ2 desktop launcher
run_timeline_demo.py     Compatibility shim to the EZ2 launcher
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

## Important Note

Legacy EZ1 code paths such as `src/`, `ui/qt_gui/`, `main.py`, and `main_qt.py` have been removed from this branch. If a remaining doc still mentions them, treat that as historical drift rather than an active app surface.
