## Cross-Machine Development

Status: active
Last reviewed: 2026-05-02

Use this guide when moving EchoZero development or runtime state between laptops.

### Canonical dev path

Code and environment:

```bash
python3 scripts/dev_bootstrap.py
.venv/bin/python run_echozero.py
```

Do not treat copied virtualenvs as the primary handoff path.
Use a fresh clone/bootstrap on the target machine, then import the minimum
machine-local runtime state.

### Portable runtime state

Portable by default:

- installed runtime models at `~/.echozero/models`
- app settings JSON from the active local settings store

Not part of the normal portable payload:

- `~/.echozero/working`
- runtime logs
- project-local generated artifacts

Why:

- models are required for local runtime bundle resolution
- settings carry machine-local app behavior such as audio, MA3 OSC, and import defaults
- working directories and logs are scratch/debug state, not baseline setup

### Export from source machine

Create an explicit handoff archive:

```bash
.venv/bin/python scripts/export_dev_state.py artifacts/dev-state/echozero-dev-state.zip
```

Notes:

- the export script uses the active app settings store path by default
- the export archive includes `manifest.json` plus the selected settings/models payloads

### Import on target machine

After cloning the repo and running bootstrap on the target laptop:

```bash
.venv/bin/python scripts/import_dev_state.py artifacts/dev-state/echozero-dev-state.zip
```

If the target machine already has models or app settings you intend to replace:

```bash
.venv/bin/python scripts/import_dev_state.py artifacts/dev-state/echozero-dev-state.zip --force
```

### Current settings-path truth

App settings are machine-local and come from the JSON settings store:

- canonical repo/dev path: `<repo>/config/app-settings.json`
- frozen app path: `<install dir>/config/app-settings.json`
- legacy fallback path: `~/.echozero/app-settings.json` on macOS/Linux or `%LOCALAPPDATA%/EchoZero/app-settings.json` on Windows

The export helper resolves the active settings store path before archiving so
legacy fallback users are still captured.

### Initial sanity check on the target machine

After import:

```bash
.venv/bin/python run_echozero.py --help
.venv/bin/python -m pytest tests/ui/test_run_echozero_launcher.py tests/test_dev_state_scripts.py -q
```

Then launch the app normally:

```bash
.venv/bin/python run_echozero.py
```
