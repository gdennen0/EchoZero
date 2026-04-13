## Real Input E2E Implementation Plan

This plan covers the true user-facing sequence:
`Open EZ -> New Project -> Load Song -> Extract Stems -> Extract Drum Events -> event actions -> MA3 actions`.

Canonical references:
- [docs/UNIFIED-IMPLEMENTATION-PLAN.md](/C:/Users/griff/EchoZero/docs/UNIFIED-IMPLEMENTATION-PLAN.md)
- [docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md](/C:/Users/griff/EchoZero/docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md)

Non-negotiable constraints carried into every phase:
- Main is truth.
- Takes are subordinate.
- MA3 sync remains main-only.
- No UI-only fake success for pipeline work.

### Phase 1
Goal: wire user-facing pipeline actions into the canonical app shell and test lanes.

Scope:
- Add timeline intents for song import and extraction actions.
- Expose inspector/context actions for add song, extract stems, and extract drum events.
- Route widget actions into runtime hooks with explicit warnings on missing capability.
- Implement canonical `add_song_from_path` in `AppShellRuntime` using `ProjectStorage.import_song`.
- Keep extraction hooks explicit and non-faking until canonical pipeline execution is ready.
- Extend Lane B DSL/runner coverage for the new action IDs.

Acceptance gate:
- User can discover and trigger the actions from the shell.
- Add-song updates the canonical presentation from persistence.
- Extraction actions fail explicitly with actionable messaging instead of silent no-op or hang.

### Phase 2
Goal: close Lane B context-menu and drag-input coverage gaps.

Scope:
- Expand Lane B scenario coverage for right-click inspector/context invocation.
- Add drag/input paths where user selection and layer targeting affect extraction staging.
- Verify new actions remain deterministic in offscreen GUI automation.

Acceptance gate:
- Lane B can drive the same action IDs from context surfaces, not just direct helpers.

### Phase 3
Goal: capture the exact real user flow in one DSL scenario and record a window-only demo.

Scope:
- Build a full DSL script for the exact sequence from app launch through MA3-facing actions.
- Keep capture limited to the app window surface.
- Require deterministic trace output plus artifact capture.

Acceptance gate:
- One repeatable scenario demonstrates the exact user-facing sequence end to end.

### Phase 4
Goal: add MA3 protocol proof hooks for the real-input flow.

Scope:
- Connect extracted event outputs to main-only MA3 push proof points.
- Add protocol assertions around payload shape and write eligibility.
- Keep non-main/take data blocked from MA3 truth writes.

Acceptance gate:
- Tests prove the real-input path reaches MA3 hooks only through main-authorized data.
