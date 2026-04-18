# UI Automation Implementation Board

_Updated: 2026-04-18_

This board is the concrete execution plan for the clean-sheet testing and
automation redesign.
It turns the strategy in
[UI Automation Plan](/Users/march/Documents/GitHub/EchoZero/docs/UI-AUTOMATION-PLAN.md:1),
[UI Automation Library Plan](/Users/march/Documents/GitHub/EchoZero/docs/UI-AUTOMATION-LIBRARY-PLAN.md:1),
[Testing Guide](/Users/march/Documents/GitHub/EchoZero/docs/TESTING.md:1),
and [App Delivery Plan](/Users/march/Documents/GitHub/EchoZero/docs/APP-DELIVERY-PLAN.md:1)
into one implementation board.

## Mission

Build one automation model and one user-equivalent control surface:

- one app-owned semantic object model
- one user-equivalent control surface over the canonical app path
- one bridge contract for read, act, and capture
- one proof ladder from contract to appflow to human-path release evidence

Everything else is migration scaffolding or deletion candidates.

## Locked Design Decisions

1. The canonical runtime path is `run_echozero.py` plus the real app shell.
2. The canonical automation model is app-owned semantic objects with declared
   facts, actions, and hit targets.
3. The canonical control surface is hybrid:
   semantic query/invoke first, pointer and keyboard execution second, OS
   fallback only for system-owned UI.
4. `packages/ui_automation` is the incubation boundary for reusable automation
   core work. EchoZero is the first adapter, not the core API shape.
5. `humanflow-all` is the broad app-path automation lane, but it is not by
   itself a human-path demo claim.
6. Packaging smoke, manual UX QA, and real MA3 hardware validation remain
   separate release gates.

## Hard Rules

The redesign must not introduce runtime backdoors.

- No hidden test-only commands that mutate app state outside the same actions an
  operator can reach through the app shell.
- No widget-state injection, presenter-only bypasses, or direct model mutation
  marketed as UI automation.
- No duplicate truth store outside the app to make automation easier.
- No image matching as the primary locator or truth model.
- No arbitrary command execution through the automation bridge.
- No public remote-control surface. Bridge exposure stays localhost, opt-in,
  capability-limited, and session-scoped.
- No alternate launcher path for automation. The automation surface rides the
  canonical app entrypoint.
- No app semantics invented in wrappers when the app contract does not expose
  them.

## Ownership Model

Use these owners on tasks:

- `AUT`: reusable automation core and bridge boundary
- `APP`: EchoZero application/runtime contract
- `QT`: Qt shell, metadata, and timeline hit-surface wiring
- `QA`: proof lanes, migration coverage, reporting
- `REL`: packaged smoke, release evidence, and operator signoff

## Proof Lanes

Use these proof lanes consistently on tasks:

- `P0 Contract`: targeted pytest for provider/session/bridge contract
- `P1 AppFlow`: `python -m echozero.testing.run --lane appflow`
- `P2 Sync`: `python -m echozero.testing.run --lane appflow-sync`
- `P3 Protocol`: `python -m echozero.testing.run --lane appflow-protocol`
- `P4 GUI Sim`: `python -m echozero.testing.run --lane gui-lane-b`
- `P5 HumanFlow`: `python -m echozero.testing.run --lane humanflow-all`
- `P6 Package`: packaged build plus packaged smoke path from
  `docs/APP-DELIVERY-PLAN.md`
- `P7 Human QA`: milestone manual walkthrough
- `P8 Hardware`: real MA3 hardware validation

## Phase Board

### Phase A. Freeze One Model

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-001 | Freeze the canonical automation contract as `objects + targets + actions + hit_targets + artifacts`. | AUT | P0 | No new flat-only or widget-private contract grows beside it. |
| UIA-002 | Freeze `packages/ui_automation` as the incubation boundary and declare EchoZero as first adapter only. | AUT | P0 | Core imports remain Qt-free and EchoZero-free. |
| UIA-003 | Freeze `run_echozero.py` as the only automation launch path. | APP | P1 | Automation attach proves the same launcher path as dev/runtime use. |
| UIA-004 | Publish the no-backdoor rule set as merge-gate policy for automation work. | APP | P1 | PRs touching automation reference these rules. |

### Phase B. Read-Only Truth First

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-101 | Add stable automation ids and accessible names to shell regions and major controls. | QT | P0, P1 | Snapshot targets stay stable across appflow runs. |
| UIA-102 | Export read-only snapshots from the real app with focus, selection, and object identity. | APP | P0, P1 | Live bridge snapshots round-trip without object loss. |
| UIA-103 | Export timeline hit targets, coordinate mapping, and object-at-point lookup from app truth. | QT | P0, P4 | Timeline control no longer depends on pixel guessing. |
| UIA-104 | Add screenshot capture at window, region, and timeline scopes. | AUT | P0, P1 | Review artifacts come from the live bridge, not helper-only code. |
| UIA-105 | Expand object coverage beyond current selection to visible layers, hovered/selected events, and transfer surfaces. | APP | P0, P1, P4 | The visible work surface is semantically inspectable. |

### Phase C. User-Equivalent Input

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-201 | Stabilize pointer primitives: move, hover, click, double-click, drag, scroll. | AUT | P0, P4, P5 | Pointer execution is target-backed and deterministic enough for proof lanes. |
| UIA-202 | Stabilize keyboard primitives: press, hotkeys, text entry, focus transitions. | AUT | P0, P1, P5 | Key flows match operator-visible behavior. |
| UIA-203 | Keep semantic invoke limited to app-declared actions already reachable in the app shell. | APP | P0, P1 | Invoke does not become a privileged side channel. |
| UIA-204 | Record whether each scenario step used semantic invoke, pointer/key execution, or OS fallback. | QA | P0, P5 | Reports can distinguish human-equivalent control from fallback control. |

### Phase D. Lifecycle And Operator Flows

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-301 | Prove live `app.new` through the bridge. | APP | P0, P1, P5 | New-project flow is stable enough for default automation use. |
| UIA-302 | Prove live `app.save_as` against deterministic temp paths. | APP | P0, P1, P5 | Save-as flow is stable without hidden state injection. |
| UIA-303 | Prove live `app.open` against saved projects. | APP | P0, P1, P5 | Open flow is stable through the canonical app path. |
| UIA-304 | Harden import and add-song flows over the bridge. | APP | P0, P1, P5 | Import is a first-class operator flow, not a harness-only helper. |
| UIA-305 | Promote only stable lifecycle/import flows into `humanflow-all`. | QA | P5 | The broad lane stays useful instead of flaky. |

### Phase E. Sync And Timeline Guardrails

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-401 | Expand timeline manipulation coverage for selection, drag, nudge, duplicate, and ruler/seek interactions. | QT | P0, P4, P5 | Hot timeline behaviors are controllable through one surface. |
| UIA-402 | Add semantic coverage for transfer surfaces and sync state without breaking main-only rules. | APP | P0, P2, P3, P5 | Automation cannot bypass sync guardrails. |
| UIA-403 | Keep follow-scroll, zoom, and visible-hit geometry provable when timeline controls change. | QT | P0, P4 | Timeline regressions are caught before release lanes. |

### Phase F. OS Fallback And Release Surfaces

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-501 | Add explicit OS fallback only for native dialogs, focus switching, and packaged-app surfaces. | AUT | P0, P5, P6 | Fallback is narrow, logged, and not the default. |
| UIA-502 | Add packaged-build automation coverage that reuses the same control model where possible. | REL | P6 | Packaged smoke is automation-aware without a second harness model. |
| UIA-503 | Keep milestone manual QA and real MA3 hardware validation as release signoff, not library proof. | REL | P7, P8 | Release gates stay app-first and operator-real. |

### Phase G. Cleanup And Demotion

| ID | Task | Owner | Proof | Exit |
|---|---|---|---|---|
| UIA-601 | Demote demo-only automation surfaces in docs and reviews once the new board phases are live. | QA | P1, P5 | Demo-only evidence no longer appears as signoff truth. |
| UIA-602 | Quarantine or remove duplicated target/action vocabularies that do not derive from app truth. | AUT | P0 | One vocabulary remains. |
| UIA-603 | Quarantine direct bridge actions that bypass app-declared actions or human-equivalent input. | APP | P0, P1 | No privileged runtime control survives migration. |

## Migration Order

Migrate in this exact order:

1. Freeze one contract and one launcher path.
2. Make the app fully inspectable before adding more control.
3. Expand object coverage before expanding action count.
4. Stabilize lifecycle flows before promoting them into `humanflow-all`.
5. Expand timeline and sync flows only after read-only truth and focus state are
   stable.
6. Add OS fallback only after app-owned control is solid.
7. Reuse the same control model for packaged smoke wherever system-owned UI does
   not force fallback.
8. Demote or quarantine old helper paths only after the replacement proof lanes
   are green.

## Delete / Quarantine List

These are the explicit cleanup targets for the redesign.
Some are immediate quarantine items; some become deletions once replacement
proof is green.

| Item | Status Target | Reason |
|---|---|---|
| `echozero/testing/e2e/adapters.py` demo-oriented driver | Quarantine | `docs/REAL-APP-AUTOMATION.md` already says it is not the primary harness. |
| Flat `targets + actions` only snapshot thinking | Delete as primary model | The board standard is object-first with targets/actions/hit targets attached. |
| Widget-private helper semantics not exposed through app truth | Quarantine then delete | They create runtime backdoors and duplicate meaning. |
| Direct state injection used to fake visible UI work | Delete | Violates human-equivalent control and demo rules. |
| Image-matching-first locator strategies | Delete as primary approach | Too brittle for Qt timeline control. |
| Alternate launcher or attach paths for automation-only startup | Delete | Conflicts with app-first acceptance and single launcher rules. |
| Any bridge verb that exists only because the UI/app contract is missing | Quarantine | Replace with app-declared action or user-equivalent input. |

## Completion Standard

The redesign is complete only when all of the following are true:

- One semantic object model is used across runtime snapshot, bridge, and tests.
- One user-equivalent control surface drives app-path automation.
- `humanflow-all` covers stable operator flows without relying on runtime
  backdoors.
- Sync and timeline changes remain provable through app-path and guardrail
  lanes.
- Packaged smoke, manual QA, and hardware validation remain separate release
  gates.
- Demo-only helper paths are demoted or quarantined in docs and review policy.
