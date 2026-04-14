# Alpha Closure Board (2026-04-13)

## Implement now
- Keep canonical app flow proof green and visible:
  - `add_song_from_path -> extract_stems -> extract_drum_events -> classify_drum_events -> event edits -> MA3 push/pull surfaces`.
  - Evidence: `tests/ui/test_app_shell_runtime_flow.py`, `tests/testing/test_gui_lane_b.py`, `tests/gui/scenarios/e2e_core.json`.
- Keep MA3 main-only guardrails enforced in transfer-plan apply paths.
  - Evidence: `echozero/application/timeline/orchestrator.py`, `tests/application/test_transfer_plan_batch_apply.py`.
- Keep appflow/package/smoke gates green on every closure tranche.
  - Evidence: `echozero/testing/run.py`, `scripts/run-appflow-gates.ps1`.

## Defer now (with rationale)
- Undo/redo stack in app-shell editing flow.
  - Rationale: important, but not required for first alpha-functional cut line.
- Derived-audio arbitrary stem reruns beyond current canonical source-song extraction path.
  - Rationale: not blocking operator proof path for this alpha.
- Setlist UX expansion.
  - Rationale: primitives exist; full UX is not required for current alpha signoff.

## Manual signoff required
- Packaged manual QA walkthrough evidence (required for release claim).
- Real MA3 hardware validation evidence (required beyond simulated/protocol confidence).

## Exact gate command bundle
```powershell
C:\Users\griff\EchoZero\.venv\Scripts\python.exe -m pytest tests\testing\test_gui_lane_b.py tests\ui\test_app_shell_runtime_flow.py -q
C:\Users\griff\EchoZero\.venv\Scripts\python.exe -m echozero.testing.run --lane appflow
C:\Users\griff\EchoZero\.venv\Scripts\python.exe -m echozero.testing.run --lane appflow-sync
C:\Users\griff\EchoZero\.venv\Scripts\python.exe -m echozero.testing.run --lane appflow-protocol
C:\Users\griff\EchoZero\.venv\Scripts\python.exe -m echozero.testing.run --lane appflow-all
powershell -File scripts\run-appflow-gates.ps1
```

## Completion checklist
- [x] [Agent] Council audits refreshed and reconciled (feature gaps, testing migration, release readiness).
- [x] [Agent] Lane B canonical path includes transfer-plan apply + visual checkpoints (`acfea0e`).
- [ ] [Agent] Capture and attach one clean operator-visible E2E artifact bundle for release notes.
- [ ] [Griff] Run packaged manual QA walkthrough and mark pass/fail with notes.
- [ ] [Griff] Run real MA3 hardware validation and record outcomes.
- [ ] [Agent] Final alpha claim only after all above signoff evidence is attached.
