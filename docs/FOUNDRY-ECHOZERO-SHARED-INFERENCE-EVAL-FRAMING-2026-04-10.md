# EchoZero + Foundry Shared Inference/Eval Framing (2026-04-10)

## Purpose

Define one shared middle layer for **inference contracts** and **evaluation contracts** so:

- Foundry can keep training/export orchestration app-specific.
- EchoZero can keep runtime/editor UX wiring app-specific.
- Both still agree on core model I/O and eval semantics.

This is a framing + scaffolding pass, not a full runtime migration.

---

## 1) Inference vs Eval (Hard Boundary)

### Inference (runtime prediction)
Inference answers: **"Given input(s), what does the model predict right now?"**

- Inputs: clip/audio features + preprocessing contract + class map/model signature context.
- Outputs: predictions + confidence payload + model/runtime metadata.
- Cadence: hot path (often repeated many times).

### Eval (quality measurement)
Eval answers: **"How good was model behavior across a labeled split/scenario?"**

- Inputs: prediction outcomes and labels across a split/run.
- Outputs: metrics (`accuracy`, `macro_f1`, etc.), optional per-class/aggregate/confusion summaries.
- Cadence: run-stage/analysis path (not hot-path inference calls).

### Rule
Inference and eval are separate contracts, but both are versioned and fingerprintable under one shared API package.

---

## 2) Shared Core + App Adapters

## Shared core modules (new scaffold)

Package: `echozero.inference_eval`

- `core.py`
  - `InferenceContract`
  - `EvalContract`
  - `InferenceRequest` / `InferenceResult`
  - `EvalRequest` / `EvalResult`
  - `InferenceCore` / `EvalCore` protocol interfaces
  - contract fingerprint helpers (`canonical_contract_payload`, `contract_fingerprint`)

This layer contains no Foundry/Qt UI side effects.

## App adapters (new scaffold)

- `echozero.inference_eval.foundry_adapter`
  - `create_foundry_adapter(...)`
  - `FoundrySharedAdapter`
  - Bridges run spec + class map into shared inference contract
  - Bridges shared eval request/result into Foundry eval payload shape

- `echozero.inference_eval.echozero_adapter`
  - `create_echozero_adapter(...)`
  - `EchoZeroSharedAdapter`
  - Bridges checkpoint/runtime metadata into shared inference contract
  - Provides compact runtime summary payload for app surfaces

Adapters own app wiring; shared core owns contract semantics.

---

## 3) No-Drift Guarantees

The new scaffold enforces no-drift behavior at the contract level:

1. **Deterministic contract fingerprints**
   - Fingerprints are derived from canonical JSON payloads with stable key sorting.
   - Equivalent contracts with different dict ordering hash identically.

2. **Copy-on-ingest contract payloads**
   - Contracts copy incoming maps into internal dicts.
   - Prevents accidental mutation coupling between caller/app state and shared contract state.

3. **Explicit contract schema IDs**
   - Inference and eval contracts each carry explicit schema version strings.

4. **Adapter translation boundaries**
   - Foundry/EchoZero translation happens in adapter modules only.
   - Core contract objects remain app-agnostic.

---

## 4) Acceptance Tests (Current)

Added lightweight tests in `tests/foundry/test_shared_inference_eval_scaffolding.py`:

- Contract fingerprint remains stable across equivalent payload ordering.
- Foundry adapter creates eval payload with expected contract fields.
- EchoZero adapter reads inference preprocessing from checkpoint metadata.

These tests validate contract/scaffold behavior without changing existing training/eval execution flows.

---

## 5) Phased Implementation Plan (Tied to Current Project State)

Current state (from tracker/docs):
- Foundry trainers + eval reports are operational.
- EchoZero runtime classification consumes checkpoint `inference_preprocessing` metadata.
- We need shared semantics now, without destabilizing active flows.

## Phase A (this change) — Framing + additive scaffold
- Add shared contract interfaces and adapter entrypoints.
- Add contract-level tests only.
- Keep all current run/train/eval production paths unchanged.

Exit criteria:
- No regressions in existing Foundry tests.
- New contract tests green.

## Phase B — Dual-write/dual-validate integration (next)
- Thread shared contracts into Foundry run/eval service boundaries as optional mirrors.
- Keep legacy payloads as source of execution truth while comparing fingerprints/logs.

Exit criteria:
- Shared and legacy payloads align on representative runs.
- Drift alerts available for mismatches.

## Phase C — Shared core as default internal representation
- Promote shared contract objects to default in both Foundry and EchoZero runtime wiring.
- Keep app adapters as the only app-specific boundary.

Exit criteria:
- Existing artifacts/eval reports remain backward-compatible.
- No UI/runtime contract regression.

## Phase D — tighten guarantees + regression packs
- Add richer acceptance coverage (artifact manifests, eval schema evolution cases, checkpoint/backfill compatibility).
- Add CI gate on shared contract fingerprint consistency for known fixtures.

Exit criteria:
- Contract drift caught pre-merge.
- Shared middle is stable and trusted.

---

## 6) Non-Goals for This Pass

- No trainer rewrites.
- No eval metric algorithm changes.
- No artifact schema version bumps.
- No migration of existing persisted records.

This is intentionally additive scaffolding for safe convergence.
