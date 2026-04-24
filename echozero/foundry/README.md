# Foundry

Status: canonical subsystem map
Last verified: 2026-04-21

Foundry is the model training and artifact workflow lane for EchoZero.
It is real product code with CLI, UI, persistence, contracts, and validation behavior.
Treat it as a supported subsystem, not as an experiment bucket.

## Start Here

- `app.py`: Foundry application façade
- `cli.py`: canonical CLI surface
- `ui/main_window.py`: canonical desktop UI surface
- `domain/entities.py`: core Foundry domain entities
- `services/*`: dataset, run, artifact, trainer, query, and validation workflows
- `persistence/repositories.py`: Foundry storage access
- `contracts/*`: exported JSON contract schemas

## Canonical Flows

- Dataset creation and ingest
- Split planning and balancing
- Run creation and background training
- Artifact export
- Compatibility validation

Reference doc:
- `docs/FOUNDRY-TRAINING.md`

## Invariants

- Exported artifacts are contract-backed and validation-backed.
- Training and artifact workflows should flow through services, not directly from UI widgets.
- Background runs should remain observable and cancelable through the supported surfaces.

## Primary Tests

- `tests/foundry/`
- `tests/foundry/test_ui_smoke.py`
- `tests/processors/test_pytorch_audio_classify_preflight.py` when classification/runtime integration changes

## Forbidden Shortcuts

- Do not bypass contract validation to make exports "work."
- Do not couple UI directly to low-level persistence details when a service exists.
- Do not treat Foundry as disposable support code.
