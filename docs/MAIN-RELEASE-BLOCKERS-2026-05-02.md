## Main Release Blockers

Status: reference
Last reviewed: 2026-05-02

This note preserves the clean-`origin/main` release findings captured before
the launcher import-path work resumed.

### Scope checked

- clean worktree from `origin/main`, not the dirty local feature branch
- canonical app-flow lane on `origin/main` commit `b617b87`
- spot check on `origin/main^` commit `efd6b82`

### Findings

1. `python -m echozero.testing.run --lane appflow` failed on clean `origin/main`.
   - `tests/testing/test_app_flow_harness.py::test_app_flow_harness_derived_audio_layers_produce_distinct_playback_output`
   - `tests/testing/test_simulated_ma3_bridge.py::test_simulated_ma3_bridge_merge_push_distinguishes_cue_refs_with_same_cue_number`

2. `pytest tests/ -x --timeout=30` also failed during collection.
   - `tests/ui/test_shared_shell_style.py`
   - import error from `tests/ui/timeline_shell_support.py` because
     `test_draw_mode_drag_dispatches_create_event_intent` was not exported

3. The same failures reproduced on `origin/main^` (`efd6b82`).
   - this did not look like a one-commit regression at the current tip

4. Packaging/smoke was not fully proven on that machine state.
   - local venv was missing packaging extras required by `scripts/verify_env.py`
   - this was separate from the repo-state test failures above

### Suggested follow-up

- repair the timeline-shell test support drift
- resolve the MA3 cue-ref normalization mismatch
- investigate whether the derived-layer playback failure is a real routing bug,
  a harness/render timing issue, or a stale assertion
