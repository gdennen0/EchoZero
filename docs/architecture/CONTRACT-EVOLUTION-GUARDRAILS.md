# Shared Inference/Eval Contract Evolution Guardrails

## Why this exists

Foundry training artifacts and EchoZero runtime checkpoints must stay compatible as each side evolves.
This document defines compatibility rules so schema growth does not silently break runtime loading.

## Compatibility policy

### 1) Runtime manifest resolution is strict

- Runtime preflight scans `*.manifest.json` beside the requested model file.
- If no manifests exist, runtime behavior is unchanged (legacy permissive path).
- If manifests exist, exactly one manifest must resolve `manifest.weightsPath` to the requested model path.
- Zero matches or multiple matches are hard errors.

**Rationale:** prevents accidental pairing of a model with the wrong artifact manifest.

### 2) Human-readable summaries are stable, structured diagnostics are additive

- Existing error summary strings remain authoritative for user-facing failures.
- Validation failures now also carry structured diagnostics (`validation_report`, `validation_diagnostics`) on raised exceptions.
- Structured payloads are additive and must not replace stable summary text.

**Rationale:** preserves existing UX and tests while enabling machine-driven triage.

### 3) Fingerprint checks are optional for legacy artifacts, strict when present

- `sharedContractFingerprint` is optional in manifests for backward compatibility.
- When present, it must match checkpoint-derived (runtime) and run-derived (Foundry) fingerprints.
- A mismatch is a hard compatibility error.

**Rationale:** allows incremental adoption without weakening drift detection.

### 4) Contract fields may grow, core invariants may not drift

Future schema additions are allowed if they do not alter existing invariants:

- Required preprocessing keys remain:
  - `sampleRate`, `maxLength`, `nFft`, `hopLength`, `nMels`, `fmax`
- Class order remains meaningful and must stay stable.
- Classification mode semantics remain explicit and validated.

If semantics change, bump schema/version and provide a migration path.

## Required test guardrails

Any contract evolution PR must keep these green:

- `tests/processors/test_pytorch_audio_classify_preflight.py`
- `tests/inference_eval/test_runtime_preflight_compatibility.py`
- `tests/inference_eval/test_foundry_echozero_parity.py`
- `tests/foundry/test_artifact_shared_fingerprint.py`
- `tests/foundry/test_train_artifact_runtime_parity.py`

## Review checklist

- [ ] Runtime summary error strings unchanged where behavior is intended to stay stable.
- [ ] Structured diagnostics updated only additively.
- [ ] Manifest resolution still enforces one-to-one model/manifest binding.
- [ ] Fingerprint logic remains backward-compatible for legacy manifests.
- [ ] Parity tests cover Foundry -> artifact -> runtime flow.
