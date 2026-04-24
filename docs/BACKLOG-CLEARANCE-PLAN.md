# EchoZero Backlog Clearance Plan

Status: active ordered backlog-clearance plan
Last verified: 2026-04-22

This plan is the ordered successor to `docs/EXECUTION-PLAN.md`.
Use `docs/STATUS.md` for current repo truth.
Use `docs/LLM-CLEANUP-BOARD.md` for campaign posture.
Use `docs/APP-DELIVERY-PLAN.md` for release gates and signoff backlog.
Use this file for the remaining cleanup, streamlining, documentation, and
release backlog that still needs to be cleared.

## Goal

Clear the remaining backlog without reopening architecture drift.

That means:

- finish the "clean up and streamline the code" pass on the remaining large
  canonical hotspots
- extend the current type and boundary discipline deeper into the still-large
  timeline, Qt, Foundry, and persistence surfaces
- shrink the support/test cognitive load where it still obscures repo truth
- reduce doc sprawl so the repo points at one ordered path instead of many
  parallel backlog hints
- close the remaining release-signoff backlog

## Hard Rules

- `run_echozero.py` remains the canonical desktop entrypoint.
- Main is truth. Takes stay subordinate history/candidate surfaces.
- Widgets must not invent truth or bypass application contracts.
- Sync stays main-only and app-boundary owned.
- Support-only and simulated surfaces stay explicitly non-canonical.
- No opportunistic feature work during Phases 1-4 unless it directly unblocks
  proof, cleanup, or release signoff.
- Every slice must land with the smallest relevant proof lane before the next
  slice begins.
- Do not tighten a guardrail before the corresponding files are small and clean
  enough to pass it.

## Current Open Backlog

As of `2026-04-21`, the remaining backlog clusters are:

1. Remaining canonical hotspot decomposition
2. Remaining type/contract cleanup in large timeline/UI/Foundry hotspots
3. Large support/test helper streamlining
4. Documentation consolidation
5. Manual packaged-app QA and real MA3 hardware signoff

## Execution List

## 1. Split `object_info_panel.py`

Status: complete (2026-04-21)

- Why first:
  it is still a medium-large canonical UI surface with strong existing proof
  coverage and a lower-risk split than `widget_canvas.py`.
- Current hotspot:
  `echozero/ui/qt/timeline/object_info_panel.py` (`280` lines after split)
- Files:
  `echozero/ui/qt/timeline/object_info_panel.py`
  new helper modules under `echozero/ui/qt/timeline/`
- Steps:
  1. Separate event preview waveform/rendering from contract text assembly.
  2. Separate action-row/settings-row rendering from panel shell wiring.
  3. Keep `ObjectInfoPanel` as the stable public root.
  4. Preserve the existing signal and import surface used by the widget shell.
- Proof:
  `./.venv/bin/python -m pytest tests/ui/test_timeline_style.py tests/ui/test_timeline_shell.py -q`
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  the public panel root is a small shell over focused preview, action, and
  formatting helpers.

## 2. Split `widget_canvas.py`

Status: complete (2026-04-21)

- Why second:
  it is the single largest remaining canonical product module and still mixes
  painting, hit-testing, drag state, and context/tooltip handling.
- Current hotspot:
  `echozero/ui/qt/timeline/widget_canvas.py` (`190` lines after split)
- Files:
  `echozero/ui/qt/timeline/widget_canvas.py`
  new helper modules under `echozero/ui/qt/timeline/`
- Steps:
  1. Separate paint/layout helpers from direct input-state handling.
  2. Separate hit-target/context-menu/tool-tip logic from drag/edit state.
  3. Keep `TimelineCanvas` as the stable public root.
  4. Add perf-aware seams so timeline paint/input changes stay measurable.
- Proof:
  `./.venv/bin/python -m pytest tests/ui/test_timeline_shell.py tests/ui/test_runtime_audio.py tests/ui/test_timeline_style.py -q`
  `./.venv/bin/python -m pytest tests/benchmarks/benchmark_timeline_phase3.py -q`
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  the canvas root no longer mixes painting, hit-testing, and interaction state
  in one file and the hot path still passes the perf guardrail.

## 3. Split `main_window_run_mixin.py`

Status: complete (2026-04-21)

- Why third:
  it is the remaining large Foundry UI run/artifact surface after the main
  window shell split.
- Current hotspot:
  `echozero/foundry/ui/main_window_run_mixin.py` (`24` lines after split)
- Files:
  `echozero/foundry/ui/main_window_run_mixin.py`
  new helper modules under `echozero/foundry/ui/`
- Steps:
  1. Separate run queue controls from artifact/export controls.
  2. Separate worker-thread lifecycle from widget construction.
  3. Keep the current public mixin seam stable for `main_window.py`.
- Proof:
  `./.venv/bin/python -m pytest tests/foundry -q`
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  the Foundry run mixin reads as a thin orchestration layer over explicit run,
  artifact, and worker helpers.

## 4. Extend Guardrails After Each Split

Status: complete (2026-04-22)

- Depends on:
  `1`, `2`, `3`
- Steps:
  1. Add new root-file size ceilings in `scripts/check_repo_hygiene.py`.
  2. Keep missing-header and public-root guardrails aligned with the new files.
  3. Add focused tests when the guardrail logic changes materially.
- Proof:
  `./.venv/bin/python scripts/check_repo_hygiene.py`
  `./.venv/bin/python -m pytest tests/test_check_repo_hygiene.py -q`
- Done when:
  the newly cleaned public roots cannot silently regress back into monoliths.

## 5. Clean Remaining Large Canonical Mixins and Services

Status: complete (2026-04-22)

- Ordered target queue:
  1. `echozero/application/timeline/orchestrator_selection_mixin.py` (`864`)
  2. `echozero/application/timeline/object_action_settings_service.py` (`811`)
  3. `echozero/persistence/session.py` (`716`)
  4. `echozero/models/provider.py` (`616`)
  5. `echozero/foundry/services/baseline_trainer.py` (`768`)
  6. `echozero/foundry/ui/main_window_workspace_mixin.py` (`621`)
  7. `echozero/ui/qt/timeline/widget_action_transfer_mixin.py` (`635`)
- Steps:
  1. Prefer concern-based splits over arbitrary chunking.
  2. Replace remaining broad `object`/`getattr` edges with small typed
     protocols or explicit helpers.
  3. Keep public entry modules explicit via direct imports or `__all__`.
  4. Add guardrails only after each slice is clean enough to hold a ceiling.
- Proof:
  targeted `mypy` for each hotspot
  targeted `pytest` slice for each hotspot
  `./.venv/bin/python -m echozero.testing.run --lane appflow` for canonical app-path changes
- Done when:
  the remaining large canonical files are either reduced materially or clearly
  justified as stable exceptions.
- Completed notes:
  `orchestrator_selection_mixin.py` now stays a thin public seam over
  `orchestrator_selection_state_mixin.py` and
  `orchestrator_event_edit_mixin.py`; `object_action_settings_service.py`
  now stays a bounded public facade over the session/copy/runtime helper
  modules; `session.py`, `provider.py`, `baseline_trainer.py`,
  `main_window_workspace_mixin.py`, and `widget_action_transfer_mixin.py`
  are now reduced public roots over concern-specific helper modules. Focused
  `mypy` lanes, targeted `pytest` slices, and the canonical `appflow` lane
  passed after the split.

## 6. Reduce the Giant Support Helpers

Status: complete (2026-04-22)

- Current hotspot queue:
  1. `tests/ui/timeline_shell_support.py` (`4386`)
  2. `tests/ui/app_shell_runtime_flow_support.py` (`2486`)
  3. `tests/persistence_support.py` (`1620`)
  4. `tests/audio_engine_support.py` (`1578`)
  5. `tests/session_support.py` (`1268`)
  6. `tests/ui/runtime_audio_support.py` (`1025`)
- Steps:
  1. Split by builder/test-behavior ownership, not line count alone.
  2. Keep the historical wrapper entrypoints stable.
  3. Prevent support-only helpers from becoming shadow product logic.
- Proof:
  targeted wrapper-path pytest
  targeted split-area pytest
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  support files stop being the largest unreadable surfaces in the repo and the
  old proof commands still work.
- Completed notes:
  the old giant support roots now stay as thin compatibility wrappers over
  behavior-owned support modules plus shared helper seams:
  `timeline_shell_support.py`,
  `app_shell_runtime_flow_support.py`,
  `persistence_support.py`,
  `audio_engine_support.py`,
  `session_support.py`, and
  `runtime_audio_support.py` are all reduced to 9-17 line wrappers while the
  new support modules keep layout/object-info/transfer, project/settings/
  pipeline/audio, persistence core/layers/round-trip/integrity, clock/layers/
  integration/regressions, session dirty/save/lifecycle/edge, and runtime
  controller/widget concerns separated. The historical wrapper-path pytest
  slices and `scripts/check_repo_hygiene.py` all passed after the split, and
  the new wrapper roots plus support modules now carry default header and
  wrapper-size guardrails.

## 7. Quarantine or Delete Remaining Non-Canonical Demo Drift

Status: pending

- Steps:
  1. Sweep support/demo surfaces for obsolete compatibility aliases.
  2. Remove or quarantine any support helper that is no longer needed by
     `run_echozero.py`, canonical tests, Foundry, packaging, or docs.
  3. Keep simulated proof clearly labeled and non-canonical.
- Proof:
  `./.venv/bin/python scripts/check_repo_hygiene.py`
  targeted support-lane pytest slices
- Done when:
  support-only modules are clearly bounded and no longer imply app truth.

## 8. Reduce Doc Sprawl and Keep Status Docs in Sync

Status: pending

- Steps:
  1. Update `docs/STATUS.md` immediate cleanup focus once items `1-4` are done.
  2. Keep `docs/LLM-CLEANUP-BOARD.md` and this plan synchronized after each
     material cleanup pass.
  3. Retire stale wording in superseded execution docs once the successor path
     is established and linked everywhere needed.
  4. Keep one current-truth map (`STATUS.md`), one ordered cleanup/backlog plan
     (this file), one cleanup campaign board (`LLM-CLEANUP-BOARD.md`), and one
     release/signoff plan (`APP-DELIVERY-PLAN.md`).
  5. Merge stale planning/audit docs into fewer current-state docs when their
     remaining value is only backlog context.
  6. Update `docs/index.md` and `AGENTS.md` whenever canonical doc entrypoints
     change.
- Proof:
  doc review plus link/reference check via `rg`
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  the repo front door points at a small, current, non-duplicated doc set.

## 9. Packaged-App Manual QA

Status: pending

- Source:
  `docs/APP-DELIVERY-PLAN.md`
- Steps:
  1. Run the packaged app path on milestone checkpoints.
  2. Capture operator-visible notes/checklists for the critical path.
  3. Keep the proof separate from simulated GUI lanes.
- Proof:
  packaged-app checklist plus notes
  packaging build and smoke logs
- Done when:
  manual UX QA is complete against the packaged path.

## 10. Real MA3 Hardware Validation

Status: pending

- Source:
  `docs/APP-DELIVERY-PLAN.md`
- Steps:
  1. Validate transfer and receive flows on real MA3 hardware.
  2. Capture what was real hardware versus simulated proof.
  3. Close the final release-signoff gap only after that evidence exists.
- Proof:
  hardware validation notes/logs plus any required follow-up fixes
- Done when:
  real MA3 validation is complete and the release backlog is empty.

## Acceptance Ladder

Use these at the end of each slice as applicable:

1. `./.venv/bin/python scripts/check_repo_hygiene.py`
2. `./.venv/bin/python scripts/check_architecture_boundaries.py`
3. `./.venv/bin/python scripts/check_canonical_launcher.py`
4. targeted `pytest`
5. targeted `mypy`
6. `./.venv/bin/python -m echozero.testing.run --lane appflow`
7. perf guardrail when a hot timeline path changed
8. packaging/manual/hardware proof when a release-signoff slice changed

## Completion Standard

This backlog is clear when:

- the remaining large canonical hotspots have been decomposed or explicitly
  justified with stable guardrails
- the remaining large support/test helpers are materially smaller and easier to
  navigate
- doc sprawl is reduced to one current-truth map, one ordered backlog plan, one
  cleanup board, and one release/signoff plan
- manual packaged-app QA is complete
- real MA3 hardware validation is complete
