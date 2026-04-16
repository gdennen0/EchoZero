# EchoZero 2 Installation

This branch now targets EZ2-only app surfaces.

Use these install and launch paths as canonical:

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want the broader dev toolchain too:

```bash
pip install -e ".[dev]"
```

## Run The EZ2 App

```bash
python run_echozero.py
```

## Run Foundry

```bash
python -m echozero.foundry.cli --root . ui
```

## Legacy Note

Legacy EZ1 app code has been removed from this branch. If you find docs that still mention `main_qt.py`, `main.py`, `src/`, or `ui/qt_gui/`, treat them as historical notes that still need cleanup.

Use [docs/EZ2-CODEBASE-CLEANUP-MAP.md](docs/EZ2-CODEBASE-CLEANUP-MAP.md) for the current keep/history map.
