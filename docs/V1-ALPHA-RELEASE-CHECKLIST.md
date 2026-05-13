# EchoZero v1-alpha Release Checklist

Status: active
Last verified: 2026-05-13

This checklist owns the v1-alpha release/install/model-distribution gate. It is intentionally narrower than architecture cleanup.

## Identity

- Release version: `1.0.0-alpha.0`
- Production app name: `EchoZero`
- Canonical desktop entrypoint: `run_echozero.py`
- Production packaging command: `python scripts/build_app.py --clean`

Keep these aligned:

- `pyproject.toml`
- `echozero/__init__.py`
- `packaging_config.json`
- `tests/test_smoke.py`

## Model Distribution

Do not bundle model weights into the app by default.

The app-managed model root is:

```text
~/.echozero/models
```

Central registry entries must provide:

- model id, type, label, version
- file URLs, sizes, SHA-256 hashes
- minimum app version
- classes
- runtime consumer
- compatibility fingerprint
- release channel

Install flow:

1. Discover a registry manifest.
2. Download files into `~/.echozero/models/.staging`.
3. Verify manifest shape, file size, SHA-256, runtime consumer, and compatibility metadata.
4. Atomically promote to a versioned model directory.
5. Update local indexes.
6. Keep app startup offline-safe.

CLI helper:

```bash
python -m echozero.models install default-drums --manifest https://example.com/echozero-models.json
python -m echozero.models set-registry https://example.com/echozero-models.json
python -m echozero.models available
python -m echozero.models list
python -m echozero.models validate
```

The Model Manager uses the same central registry source. It must show available
registry entries as missing, ready, outdated, or invalid, and it must install or
update selected entries through the staged checksum-verified path.

## Release Gates

Run in a Python 3.11+ environment:

```bash
python scripts/verify_env.py --quiet --build
pytest tests/test_smoke.py -q
python scripts/check_canonical_launcher.py
pytest tests/ui/test_run_echozero_launcher.py -q
pytest tests/test_model_distribution.py tests/application/test_object_action_model_picker_options.py -q
python -m echozero.testing.run --lane appflow
python -m echozero.testing.run --lane ui-automation
python -m echozero.testing.run --lane humanflow-all
python scripts/build_app.py --clean
```

Packaged smoke must produce a `smoke-report.json` with a pass result and prove:

- app launches without installed models
- model-backed actions show missing-model state instead of crashing
- Model Manager can discover registry entries and install/update a selected entry
- a local or registry model install is discovered after app restart
- packaged app does not include large `.pth` weights by default

## Rollback Triggers

Stop the alpha release on:

- packaged launch crash
- auth/config failure that blocks first run
- model install/update regression
- data-loss or persistence issue
- unintended sync/write behavior
- packaged app requiring bundled model weights to start
