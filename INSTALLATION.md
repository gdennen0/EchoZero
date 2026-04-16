# EchoZero 2 Installation

This branch now targets EZ2-only app surfaces.

Use these install and launch paths as canonical:

## Install

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

EchoZero requires Python 3.11 or newer. If your Mac still resolves `python3`
to 3.9, install a newer Python and use that exact binary when creating the venv.
The quickest macOS path is usually `brew install python@3.11`.

If you want the broader dev toolchain too:

```bash
pip install -e ".[dev]"
```

If you want the full local environment, including optional ML processors and packaging tools:

```bash
pip install -r requirements.txt
```

## Run The EZ2 App

```bash
python run_echozero.py
```

## Run Foundry

```bash
pip install -e ".[ml]"
python -m echozero.foundry.cli --root . ui
```

## Legacy Note

Legacy EZ1 app code has been removed from this branch. If you find docs that still mention `main_qt.py`, `main.py`, `src/`, or `ui/qt_gui/`, treat them as historical notes that still need cleanup.

Use [docs/EZ2-CODEBASE-CLEANUP-MAP.md](docs/EZ2-CODEBASE-CLEANUP-MAP.md) for the current keep/history map.
