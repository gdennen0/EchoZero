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

After uploading macOS assets to GitHub, download them back and verify the exact
zip(s) before asking anyone to test:

```bash
python scripts/verify_macos_release_artifact.py ~/Downloads/EchoZero-macOS.zip \
  --expected-sha256 b761a78ae276762402e62d85632727cd4245641622abea719936ce71a7e7c7ce \
  --expected-binary-uuid AC4A6A20-9E69-F717-9BBE-F9025FA69EB7 \
  --compare-zip ~/Downloads/EchoZero-v1.0.0-alpha.0-macos-arm64.zip
```

Current approved alpha macOS facts:

- main/tag commit: `bbef89e937a8165cab8cbb359b625431fb0432c8`
- tag: `v1.0.0-alpha.0`
- zip SHA-256: `b761a78ae276762402e62d85632727cd4245641622abea719936ce71a7e7c7ce`
- binary UUID: `AC4A6A20-9E69-F717-9BBE-F9025FA69EB7`
- rejected stale/bad UUIDs: `5081D799-579B-064C-8AA6-A16866024922`,
  `62C459E2-BA4A-AB28-794C-4C90ACCCA8D6`

Packaged smoke and macOS zip verification must produce pass reports and prove:

- app launches without installed models
- model-backed actions show missing-model state instead of crashing
- Model Manager can discover registry entries and install/update a selected entry
- a local or registry model install is discovered after app restart
- packaged app does not include large `.pth` weights by default
- downloaded macOS zip SHA-256 matches the approved value
- `Contents/MacOS/EchoZero` has the approved Mach-O UUID, not a stale bad UUID
- strict codesign verification passes after extraction from the downloaded zip
- smoke launch does not create `Contents/MacOS/config/app-settings.json` or any
  other mutable config file inside the app bundle
- convenience and versioned macOS zips contain byte-equivalent app payloads when
  both names are published

## Rollback Triggers

Stop the alpha release on:

- packaged launch crash
- GitHub-downloaded macOS zip verification failure
- auth/config failure that blocks first run
- model install/update regression
- data-loss or persistence issue
- unintended sync/write behavior
- packaged app requiring bundled model weights to start
