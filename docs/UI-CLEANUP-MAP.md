# UI Cleanup Map

Status: reference
Last reviewed: 2026-04-30



This file is the working cleanup and hardening map for the EchoZero UI stack.
It exists to separate the reusable Stage Zero shell from demo/dev-only surfaces
and to keep UI truth aligned with the application contracts.
It connects the current UI audit to concrete preserve/extract/delete/prove-next
steps.

## Goal

Keep the canonical UI path centered on:

- `run_echozero.py`
- `echozero/ui/qt/app_shell.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/**`
- `echozero/application/presentation/**`
- `echozero/ui/FEEL.py`
- `echozero/ui/style/**`

Move demo/dev-only helpers out of canonical runtime paths.
Remove UI behavior that still carries older take-truth or EZ1 mental models.

## Current Read

The current UI stack has three useful layers:

1. **Canonical shell**
   - `run_echozero.py`
   - `echozero/ui/qt/app_shell.py`
   - `echozero/ui/qt/timeline/widget.py`

2. **Reusable UI building blocks**
   - `echozero/ui/qt/timeline/blocks/**`
   - `echozero/application/presentation/models.py`
   - `echozero/application/presentation/inspector_contract.py`
   - `echozero/ui/FEEL.py`
   - `echozero/ui/style/**`

3. **Demo / dev-loop helpers**
   - `echozero/ui/qt/timeline/demo_app.py`
   - `echozero/ui/qt/timeline/demo_walkthrough.py`
   - `echozero/ui/qt/timeline/test_harness.py`
   - `echozero/ui/qt/timeline/fixture_loader.py`
   - `echozero/ui/qt/timeline/real_data_fixture.py`
   - `echozero/ui/qt/timeline/drum_classifier_preview.py`

The reusable layer is strong.
The canonical shell still mixes in too much workflow and demo/dev concern.

## Preserve

Keep these as the canonical and reusable UI surfaces:

- `run_echozero.py` as the only primary desktop entrypoint
- `echozero/ui/qt/app_shell.py` as the runtime/storage adapter
- `echozero/ui/qt/timeline/widget.py` as the Stage Zero shell root
- `echozero/ui/qt/timeline/blocks/**` as the paint/layout primitive set
- `echozero/application/presentation/models.py`
- `echozero/application/presentation/inspector_contract.py`
- `echozero/ui/FEEL.py`
- `echozero/ui/style/tokens.py`
- `echozero/ui/style/scales.py`
- `echozero/ui/style/qt/qss.py`
- `echozero/ui/qt/timeline/style.py`

## Extract

These should move out of the canonical runtime path or be split into cleaner seams:

1. `echozero/ui/qt/timeline/widget.py`
   - extract manual-pull dialog/UI
   - extract object-info panel construction
   - extract pure geometry/label helpers
   - reduce stringly contract-action routing

2. `echozero/ui/qt/app_shell.py`
   - remove demo/fixture side-effect registration from import-time behavior
   - keep launcher/runtime composition separate from presentation fallback policy
   - stop inferring truth from titles, badges, or status labels

3. `echozero/ui/qt/timeline/demo_app.py`
   - keep reusable demo-runtime helpers if still needed
   - separate CLI/capture/demo-runner behavior from reusable helpers

## Delete Or Quarantine

These are not canonical runtime surfaces and should be deleted or moved to dev/test-only territory:

- `echozero/ui/qt/timeline/demo_walkthrough.py`
- CLI/demo-runner code in `echozero/ui/qt/timeline/demo_app.py` once no longer needed
- `echozero/ui/qt/timeline/drum_classifier_preview.py` unless promoted into an explicit supported tool
- runtime-package dependence on `fixture_loader.py`, `real_data_fixture.py`, and `test_harness.py` unless a real app path still needs them

## Divergence To Correct

The main architectural drifts to correct are:

- `TimelineWidget` still acts as a workflow controller rather than a thin view
- `AppShellRuntime` still mixes composition-root work with truth/presentation assembly
- runtime UI still exposes take-selection and take-local action affordances that drift from the “main is truth” model
- FEEL/layout ownership is split between FEEL, style tokens, layout helpers, and widget-local geometry

## Proof Next

Strengthen proof in this order:

1. canonical launcher smoke on `run_echozero.py` without demo path
2. app-shell runtime flow:
   - import song
   - extract stems
   - save/reopen
   - derived-layer guardrails
3. timeline shell contract tests:
   - take expansion
   - selection
   - mute/solo
   - transfer preview/apply/cancel
   - follow-scroll
4. FEEL/style contract tests to keep geometry centralized
5. real-data fixture proof for expanded takes and rendering alignment

## Working Rule

If a UI file is not needed by:

- `run_echozero.py`
- the app shell runtime
- Stage Zero timeline shell
- Foundry shell
- active UI proof lanes

then it should default to extraction, dev-only quarantine, or removal rather
than staying in the canonical runtime path.
