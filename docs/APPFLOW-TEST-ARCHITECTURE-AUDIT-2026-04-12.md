# AppFlow Test Architecture Audit

Date: 2026-04-12

## Scope

Audited the current app-flow testing foundation centered on `echozero/testing/app_flow.py`, `echozero/testing/ma3/simulator.py`, `echozero/testing/run.py`, and the current app-shell sync wiring in `echozero/ui/qt/app_shell.py` and `echozero/application/sync/adapters.py`.

## Findings

| Area | Result | Notes |
| --- | --- | --- |
| First principles: main truth | PASS | The harness boots the canonical `AppShellRuntime`, which rebuilds presentation from project-native storage instead of inventing a parallel testing state model. |
| First principles: takes subordinate | PASS | App-flow tests exercise user intents and runtime presentation updates; they do not introduce take-level authority or alternate edit orchestration in the harness. |
| First principles: MA3 main-only boundary | PASS with gap | Current sync testing keeps MA3 behind `MA3SyncBridge` callbacks, which preserves the boundary. The gap was that OSC behavior had no built-in localhost simulation path, so transport-level verification depended on ad-hoc future glue. |
| App-first testability: native harness vs ad-hoc glue | PASS | `AppFlowHarness` already wraps the real `AppShellRuntime`, launcher controller, and widget lifecycle. That is the correct seam. |
| App-first testability: built-in MA3 transport prep | FAIL before this slice, PASS after this slice | Before this change, the foundation only simulated connect/disconnect callbacks. Tier B now adds a built-in OSC loopback service under `echozero/testing/ma3/` and optional harness lifecycle management for it. |
| Cohesion and anti-spaghetti risk in runtime/harness coupling | PASS with caution | Coupling is currently narrow: the harness owns UI bootstrapping and delegates runtime behavior to production code. The main caution is monkey-patching launcher path choosers; acceptable for tests, but that seam should stay local to the harness. |
| Runtime/harness/network boundary | PASS | The new OSC loopback stays localhost-only and remains a testing utility, not runtime logic. No MA3 transport details leaked into app runtime construction beyond optional harness helpers. |

## Assessment

The current foundation is directionally correct because it is app-first: tests boot the actual shell runtime, issue intents through the same surface users exercise, and keep MA3 concerns behind a minimal adapter contract. The main architectural hole was the absence of a built-in deterministic OSC transport simulator. Without that, Tier B would have drifted toward one-off socket setup inside tests, which is exactly how spaghetti begins.

With the Tier B loopback added as a testing utility and optional harness capability, the foundation now has a cleaner progression:

- Tier A: app/runtime boot, launcher actions, sync callback semantics
- Tier B: deterministic localhost OSC transport simulation
- Tier C: higher-level MA3 protocol and end-to-end sync scenarios on top of the same harness

## Recommended Next Steps

### Tier B

1. Keep `OSCLoopback` focused on transport concerns only: lifecycle, send/capture, wait/clear, localhost binding.
2. Add protocol-level fixtures next, not inside the harness. The harness should expose helpers, while MA3 message shapes live in dedicated test builders or fixtures.
3. Start validating canonical MA3 paths and argument shapes against the loopback before introducing any richer fake console behavior.

### Tier C

1. Introduce a thin MA3 protocol adapter test layer that serializes real app intents into OSC messages and parses inbound OSC into app-facing sync events.
2. Drive Tier C through app-flow scenarios, not direct service pokes, so the main truth remains the app runtime.
3. Add divergence and reconnect scenarios only after protocol fixtures exist; do not bake retry logic into the harness.
4. Keep the MA3 boundary main-only by ensuring takes and timeline editing remain validated through app intents and presentation state, while MA3 stays an external synchronized target.
