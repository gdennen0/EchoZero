# EchoZero Project Plan (Canonical)

Status: active
Last verified: 2026-05-03



This is the single source-of-truth project plan for current EchoZero development.

It replaces the old "implementation-plan-only" posture with one canonical plan
that covers:

- product truth and architectural constraints
- current repo status and delivery posture
- active workstreams
- strict execution order
- acceptance gates for reopening feature growth

If another plan doc conflicts with this file, this file wins.
Feature-specific execution plans remain valid only inside their own scope and
must not override the order or constraints here.
The current repo-wide execution order lives in
`docs/CANONICAL-EXECUTION-PLAN.md`.

---

## 1) Project Goal

Ship EchoZero as a real app-first timeline and MA3 workflow product with:

- one canonical desktop path
- one canonical truth model
- one guarded sync boundary
- one reliable proof stack
- enough structural cleanup to support future feature growth without drift

The current program goal is not net-new feature expansion.
The current program goal is to finish signoff-critical proof and close the
remaining architecture debt that still threatens delivery quality.

---

## 2) Success Criteria

EchoZero is considered ready to reopen broader feature work only when all of
the following are true:

1. The canonical app path is the acceptance truth for playback, transfer, and
   sync behavior.
2. Main remains the only live truth lane for playback/export/sync.
3. MA3 push and pull behavior is guarded by app-boundary proof, not widget-only
   behavior.
4. Packaged-app smoke, app-flow, and sync proof lanes remain green.
5. Manual packaged UX walkthrough and real MA3 hardware validation are complete.
6. The highest-risk canonical files are reduced enough that future changes do
   not depend on monolithic app-shell or orchestrator edits.

---

## 3) Source-of-Truth Order (Highest -> Lowest)

1. Original distillation intent
   - upstream authority: `memory/echozero-distillation/DISTILLATION.md`
   - branch interpretation is captured in `docs/architecture/DECISIONS.md` and
     this file
2. First principles
   - `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
3. Current repo truth
   - `docs/STATUS.md`
4. Application and sync contract
   - `echozero/application/timeline/*`
   - `echozero/application/sync/*`
   - `echozero/infrastructure/sync/ma3_adapter.py`
5. App-first delivery contract
   - `docs/APP-DELIVERY-PLAN.md`
6. FEEL contract
   - `echozero/ui/FEEL.py`
7. Evidence and proof lanes
   - `tests/application/**`
   - `tests/ui/**`
   - `tests/testing/**`

---

## 4) Locked Decisions (Non-Negotiable)

### Truth model

- Main is truth.
- Takes are subordinate candidates/history/comparison inputs, never alternate
  live truth.
- No active-take truth model returns in app or UI.

### Application boundary

- Engine is app-agnostic and returns typed outputs.
- Application owns truth mutation, freshness, provenance, and layer/take
  mapping.
- Widgets must not invent alternate truth or bypass the application contract.

### Sync boundary

- MA3 sync is main-only.
- Non-main data does not become MA3 truth directly.
- Sync writes fail hard on malformed required event metadata.

### Versioning and persistence

- New `SongVersion` starts blank.
- Configs carry forward.
- Processed outputs do not carry forward automatically.

### FEEL ownership

- `echozero/ui/FEEL.py` owns UI tuning constants.
- FEEL does not own truth, persistence, or sync semantics.

### Hygiene

- Generated/runtime outputs are not tracked in git.
- Do not reintroduce alternate live-truth paths, legacy launcher surfaces, or
  widget-only workflow logic.

---

## 5) Current State

### Proven now

- `run_echozero.py` is the canonical desktop launcher.
- The app shell runtime, launcher commands, and packaged smoke path exist.
- App-flow lanes cover the canonical shell path.
- Main-only sync guardrails are automated and green.
- MA3 simulator, OSC loopback, and receive-protocol proof exist in the repo.

### Still open

- Manual packaged UX walkthrough for milestone signoff.
- Real MA3 hardware validation for final operator trust.
- One visible operator end-to-end proof sequence for release confidence.
- Sync surface consolidation so the app contract is cleaner and less split.
- Hot-file decomposition on canonical app and timeline surfaces.

### Current risk posture

- Delivery risk is no longer "missing architecture direction."
- Delivery risk is now concentrated in:
  - remaining signoff evidence that cannot be faked
  - sync/app boundary drift during cleanup
  - oversized canonical files that make regressions easier to introduce

---

## 6) Active Workstreams

### WS1 - Alpha signoff and app-first proof

Status: active

Goal:
- close the remaining non-negotiable human-path and release-signoff proof

Scope:
- packaged manual UX walkthrough
- real MA3 hardware validation
- visible operator end-to-end proof sequence

Done when:
- packaged walkthrough notes exist
- real MA3 validation result is captured
- operator proof sequence is run on the canonical app path

### WS2 - Sync boundary consolidation

Status: active

Goal:
- reduce split-brain between sync contract, app path, and UI routing

Scope:
- define one explicit adapter seam from app contract to concrete sync behavior
- move sync assertions to app-boundary tests where they still live too low
- remove duplicated implicit sync rules from UI-facing paths

Done when:
- one documented sync boundary exists
- sync guardrails are enforced from the canonical app path
- widget internals are no longer carrying behavior truth

### WS3 - Sync receive protocol hardening

Status: active

Goal:
- prove EZ2 reliably receives and parses current MA3 plugin payloads

Scope:
- protocol fixtures for trackgroups, tracks, events, and change payloads
- parser robustness for embedded delimiters and nested payload structures
- receive-path compatibility from communication service through sync entrypoints

Done when:
- sync-receive lane is green and required
- current plugin payload shapes are covered
- parser ambiguity for current protocol is removed

### WS4 - Canonical hot-file decomposition

Status: active

Goal:
- lower regression risk on the most overloaded app and timeline surfaces

Scope:
- split `echozero/ui/qt/app_shell.py`
- split `echozero/application/timeline/orchestrator.py`
- split `echozero/application/presentation/inspector_contract.py`
- split `echozero/application/timeline/object_action_settings_service.py`
- tighten types on canonical app/UI paths

Done when:
- hot files are meaningfully smaller with preserved behavior
- proof lanes stay green through the decomposition
- future feature slices can land without broad incidental edits

### WS5 - Feature expansion reopen

Status: blocked on WS1-WS4 exit criteria

Goal:
- resume feature growth without reintroducing truth or sync drift

Rule:
- no net-new feature lane becomes active until the earlier workstreams are green

---

## 7) Execution Order (Strict)

## Phase 0 - Keep proof green while work proceeds

Goal:
- preserve the current safety baseline during all cleanup and signoff work

Required:
- keep contract lanes green
- keep app-flow lanes green
- keep packaged smoke runnable
- do not trade away proof quality for speed

## Phase 1 - Close alpha signoff proof

Goal:
- finish the remaining app-first proof that still depends on real operators or
  real hardware

Tasks:
1. Run packaged manual UX walkthrough on milestone checkpoints.
2. Validate push, pull, and receive behavior against real MA3 hardware.
3. Record one visible operator end-to-end proof sequence through the canonical
   app path.

Exit criteria:
- WS1 complete

## Phase 2 - Consolidate the sync surface

Goal:
- remove app-path and widget-path ambiguity around sync behavior

Tasks:
1. Finalize the adapter seam from sync contract to concrete implementation.
2. Push behavior assertions upward to the app boundary.
3. Eliminate remaining UI-local sync rule duplication.

Exit criteria:
- WS2 complete

## Phase 3 - Harden sync receive protocol

Goal:
- make receive-path correctness cheap to prove and hard to regress

Tasks:
1. Lock current payload fixtures into protocol tests.
2. Cover parser edge cases for nested and delimiter-heavy payloads.
3. Keep the dedicated receive lane required before sync-affecting merges.

Exit criteria:
- WS3 complete

## Phase 4 - Decompose canonical hot files

Goal:
- reduce change risk in the files most likely to absorb accidental behavior

Tasks:
1. Split the app shell along stable runtime/lifecycle boundaries.
2. Split the timeline orchestrator by intent family without changing truth
   rules.
3. Split presentation and object-action contract surfaces where the current
   files obscure responsibility.
4. Tighten types while preserving the canonical app path.

Exit criteria:
- WS4 complete

## Phase 5 - Reopen feature expansion

Prerequisites:
- WS1 through WS4 complete
- acceptance gates below are green

Rule:
- every new feature maps to an explicit decision/principle and required proof
  lane before implementation starts

---

## 8) Acceptance Gates

### Gate A - Truth integrity

- no active-take truth resurrection
- no UI-only semantic state pretending to be truth

### Gate B - Sync safety

- main-only writes remain provable in tests
- missing or invalid metadata fails hard on MA3 write paths

### Gate C - App-first proof

- main app path is the required acceptance path
- packaged smoke and app-flow lanes remain green
- demo-only proof is never release signoff

### Gate D - Protocol evidence

- sync receive lane stays green and required
- current MA3 payload shapes are fixture-backed

### Gate E - Structural safety

- decomposition does not widen scope into behavior redesign
- hot-path changes carry perf guardrails where relevant

### Gate F - Repo hygiene

- no generated/runtime files tracked
- hygiene and CI checks pass

---

## 9) Risks and Mitigations

1. Real MA3 validation can expose behavior not covered by simulator fixtures.
   - Mitigation: keep protocol fixtures editable and push discoveries into
     receive and app-flow lanes immediately.
2. Hot-file decomposition can accidentally become behavior redesign.
   - Mitigation: require small slices, app-path proof, and no opportunistic
     cleanup outside owned paths.
3. Manual signoff work can drift because it is less automatable than test
   lanes.
   - Mitigation: treat manual proof as a tracked workstream with explicit exit
     criteria, not as background intent.

---

## 10) Working Rule

Build order is:

1. preserve truth
2. preserve app-path proof
3. close signoff evidence
4. consolidate sync boundary
5. harden receive protocol
6. reduce structural risk
7. reopen feature growth

Do not reverse this order for convenience.
