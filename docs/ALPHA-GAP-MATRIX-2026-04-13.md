# Alpha Gap Matrix (2026-04-13)

This matrix reconciles the originally specified app flow against the current implementation on 2026-04-13 and sets the alpha cut line.

Source set used for this pass:
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/GUI-INPUT-DEMO-PRD.md`
- `docs/GUI-INPUT-DEMO-CHECKLIST.md`
- `docs/REAL-INPUT-E2E-IMPLEMENTATION-PLAN.md`
- `docs/APP-DELIVERY-PLAN.md`
- `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`
- `SPEC.md`

Note: the task referenced `memory/echozero-distillation/DISTILLATION.md`, but that workspace copy was not present during this pass. The conformance audit and `SPEC.md` were used as the local proxy.

## Executive read

Highest alpha blockers in the requested real flow:
1. Packaged manual QA walkthrough evidence is still required for an honest alpha-functional claim.
2. Real MA3 hardware validation evidence is still required beyond simulated/protocol confidence.
3. One clean visible operator E2E proof sequence must be captured and attached to release notes.
4. Release docs/tracker alignment must stay explicit so gate status is unambiguous.
5. Undo/redo remains deferred and is out of this alpha cut line.

## Matrix

| Feature area | Spec / design intent | Current status | Evidence | Alpha decision | Rationale |
|---|---|---|---|---|---|
| Project lifecycle | Canonical app path must support new/open/save/reopen through app shell | Implemented | `EchoZero/ui/qt/app_shell.py`; `run_echozero.py`; `tests/ui/test_app_shell_runtime_flow.py`; `tests/ui/test_run_echozero_launcher.py` | Implement now | Already part of the canonical path and green; keep as alpha baseline, no further tranche needed beyond regression coverage. |
| Pipeline interactions: add song | Real user flow requires `Open EZ -> New Project -> Load Song` through the canonical runtime | Implemented | `AppShellRuntime.add_song_from_path` in `EchoZero/ui/qt/app_shell.py`; `tests/ui/test_app_shell_runtime_flow.py::test_app_shell_runtime_add_song_from_path_updates_presentation` | Implement now | This is the entry point for all downstream alpha work and is already working. |
| Pipeline interactions: extract stems | Real user flow requires canonical runtime execution, not fake success | Partial | `AppShellRuntime.extract_stems` in `EchoZero/ui/qt/app_shell.py`; `tests/ui/test_app_shell_runtime_flow.py::test_app_shell_runtime_extract_stems_persists_audio_layers_and_takes`; `tests/ui/test_app_shell_runtime_flow.py::test_app_shell_runtime_extract_stems_from_derived_audio_layer_is_deferred` | Implement now | Highest user-visible gap in the requested flow. This pass closes the source-song extraction path but intentionally defers arbitrary derived-audio reruns. |
| Pipeline interactions: extract drum events | Real user flow requires `Extract Drum Events` after stems | Implemented | `echozero/ui/qt/app_shell.py`; `tests/ui/test_app_shell_runtime_flow.py`; commit `0b063a6` | Implement now | Closed as a runtime blocker; keep regression-covered in canonical app-shell flow. |
| Timeline editing actions | Selection, nudge, duplicate, drag, mute/solo, gain, transport, push/pull surfaces must be reachable from the timeline shell | Implemented | `EchoZero/ui/qt/timeline/widget.py`; `EchoZero/application/presentation/inspector_contract.py`; `tests/ui/test_timeline_shell.py` | Implement now | Already good enough for alpha demoability; maintain with regression tests only. |
| Undo/redo | Spec and architecture call for undoable editing operations with a real stack | Missing | `SPEC.md` undo sections; `docs/architecture/DECISIONS.md`; `EchoZero/editor/pipeline.py` still contains `# TODO: push to undo stack when it exists`; no app-shell undo/redo tests | Defer | Important, but not required to prove the requested alpha user flow in one slice. Add after pipeline flow and Lane B are green. |
| MA3 push/pull/sync guardrails | Manual transfer path and main-only sync boundary must be provable | Partial | `tests/application/test_manual_transfer_push_flow.py`; `tests/application/test_manual_transfer_pull_flow.py`; `tests/unit/test_ma3_communication_service_protocol.py`; `tests/unit/test_ma3_receive_path_integration.py`; `docs/UNIFIED-IMPLEMENTATION-PLAN.md` (A6 closed, release signoff pending) | Implement now | Guardrails are substantial and automated proof is closed; remaining closure is packaged/manual QA plus real MA3 hardware evidence. |
| Demo / test automation lanes | Lane B should drive the same real input path and produce trace artifacts; appflow lanes should stay green | Partial | `echozero/testing/gui_lane_b.py`; `echozero/testing/gui_dsl.py`; `tests/testing/test_gui_lane_b.py`; `tests/gui/scenarios/e2e_core.json`; commits `4076a4b`, `acfea0e`; `docs/APPFLOW-TOTAL-SUMMARY-2026-04-13.md` | Implement now | Canonical Lane B path now includes transfer-plan apply and visual checkpoints; remaining closure is release-signoff evidence capture. |
| Packaging / distribution | Canonical build and packaged smoke path must exist and remain deterministic | Partial | `scripts/build-test-release.ps1`; `scripts/smoke-test-release.ps1`; `scripts/run-appflow-gates.ps1`; `docs/APP-DELIVERY-PLAN.md`; `docs/APPFLOW-TOTAL-SUMMARY-2026-04-13.md` | Implement now | Packaging is present; remaining work is release-readiness proof, not foundational plumbing. |
| Foundry / train readiness | Optional unless needed for the alpha app promise | Partial, out of primary alpha path | `EchoZero/foundry/*`; `docs/FOUNDRY-TRAINING.md` | Defer | Foundry is real and active, but it is not on the critical path for the requested app-shell alpha flow. Keep isolated from this alpha cut line. |

## What changed in this pass

- Tranche 1 kickoff landed real source-song stem extraction in the canonical app-shell runtime.
- The new runtime path persists derived audio layers and takes instead of returning a placeholder failure.
- The current defer boundary is explicit: rerunning stem extraction from derived audio layers remains intentionally unsupported until arbitrary-layer pipeline input is wired.

## Immediate alpha recommendation

Use this cut line:
- Keep project lifecycle, timeline editing, manual transfer surfaces, packaging, and appflow lanes as the current stable baseline.
- Treat `Extract Drum Events`, Lane B real-input completion, and sync end-to-end hardening as the next alpha readiness gates.
- Keep undo/redo and Foundry expansion out of the first alpha gate unless scope changes materially.
