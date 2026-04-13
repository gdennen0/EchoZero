# Demo Suite Audit And Plan

Date: 2026-04-13

## Brief audit

What works:
- The current demo suite already produces a stable manifest and summary bundle.
- The major demo scenarios are valuable and cover both app-shell and timeline-focused surfaces.
- The PowerShell wrapper already stages a reproducible end-to-end run and mirrors artifacts into workspace temp.

Pain points:
- `echozero/testing/demo_suite.py` mixes CLI parsing, manifest writing, scenario registry concerns, and all scenario implementations in one file.
- Scenarios are not independently addressable from the CLI, which makes partial execution and maintenance slower than necessary.
- Script ergonomics are biased toward the full run, so single-scenario iteration is more manual than it should be.
- Selection behavior was not isolated by deterministic tests, which made refactoring riskier than needed.

## Plan

Phase 1. Separate scenario architecture
- [x] Move scenario implementations into `echozero/testing/demo_suite_scenarios.py`.
- [x] Publish ordered registry entries via `SCENARIO_ORDER` and `SCENARIO_RUNNERS`.
- [x] Keep `echozero/testing/demo_suite.py` focused on orchestration, manifest generation, and CLI entry.

Phase 2. Add selection-oriented CLI
- [x] Add repeatable `--scenario <name>`.
- [x] Add `--list-scenarios`.
- [x] Preserve default behavior by running the full ordered suite when no scenario is specified.
- [x] Preserve manifest structure while allowing subset execution.

Phase 3. Improve script ergonomics
- [x] Update `scripts/run-full-demo-suite.ps1` to accept `-Scenarios`.
- [x] Add `scripts/run-demo-scenario.ps1` for single-scenario execution.
- [x] Keep optional `-AudioPath` passthrough for real-data usage.

Phase 4. Deterministic validation coverage
- [x] Keep `tests/testing/test_demo_suite_manifest.py` passing.
- [x] Add selection tests that monkeypatch the registry instead of touching GUI code.
- [x] Cover list output, selected execution, and default ordered execution.

Phase 5. Follow-up maintenance
- [ ] Consider moving shared artifact-copy helpers into a smaller internal utility module if the scenario set grows further.
- [ ] Consider adding a richer scenario description surface for `--list-scenarios` if operators need human-readable labels.
- [ ] Consider a targeted smoke fixture for the PowerShell wrappers if script-level regressions become common.

## Execution status for this task

Completed in this execution:
- Scenario split and registry export.
- CLI scenario listing and selection.
- PowerShell scenario helpers.
- Deterministic selection tests.
- Validation run and targeted script runs required by the task spec.

Not changed intentionally:
- Manifest schema shape.
- Existing scenario artifact naming.
- Default full-suite execution order.
