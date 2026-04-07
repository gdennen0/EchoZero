# Foundry Full Scope Integration Summary (2026-04-07)

## Scope Completion

All requested scope items were implemented in production-safe increments with milestone commits:

1. **CRNN backend + resolver wiring**
   - Commit: `830d051`
   - Added `CrnnTrainer`, wired `model.type='crnn'` in backend factory, exported service, and added lifecycle/backend tests.

2. **Strengthened CNN/CRNN training config/stability path**
   - Commit: `057a576`
   - Added stability guardrails:
     - non-finite loss protection (CNN/CRNN)
     - gradient clipping support (`training.gradientClipNorm`)
     - weight decay support (`training.weightDecay`)
     - early-stop controls honored in CNN/CRNN (`earlyStoppingPatience`, `minEpochs`)
   - Added safer checkpoint file writes (temp + atomic replace).
   - Kept baseline path as fallback and preserved run lifecycle event names.

3. **Live telemetry files per run + dashboard live integration + failure banner**
   - Commit: `ea49fb4`
   - Added per-run telemetry outputs:
     - `foundry/runs/<run_id>/telemetry.jsonl`
     - `foundry/runs/<run_id>/telemetry.latest.json`
   - Telemetry includes epoch/loss/ETA and host stats:
     - CPU %, RAM %, RAM MB
     - GPU VRAM used/total when CUDA available
   - Dashboard builder now surfaces telemetry + failure error payload.
   - Dashboard HTML now has polling live panel (5s refresh) and failure banner.

4. **Notification cadence + anti-spam dedupe/cooldown**
   - Commit: `ea8a67c`
   - Implemented cadence in service layer:
     - start notification
     - every 3 terminal runs milestone
     - first failure notification
     - final digest when queue drains
   - Added persisted dedupe/cooldown state:
     - `foundry/tracking/notification_state.json`

5. **Batch runner v2 (CNN/CRNN) + clean summary**
   - Commit: `f701157`
   - Added/updated `scripts/run_beefy_batch.py` as v2 flow:
     - uses CNN/CRNN mix
     - stronger schedule defaults and guardrail fields
     - validates artifact compatibility per run
     - emits clean final summary payload

## Validation

### Foundry suite
- Command: `.venv\Scripts\python -m pytest tests/foundry`
- Result: **54 passed**

### End-to-end CNN + CRNN runs with artifact compatibility
Executed local e2e validation run pair:

- Workspace root: `C:\Users\griff\EchoZero\.foundry-test-tmp\foundry-e2e-yljctjzd`
- Dataset version: `dsv_de73ad158799`

Runs:
- CNN
  - run_id: `run_71d1dc93a02c`
  - artifact_id: `art_0a6b2acf2c8d`
  - status: `completed`
  - compatibility: `ok=true`
- CRNN
  - run_id: `run_9bc14e573a07`
  - artifact_id: `art_29446ad19a0f`
  - status: `completed`
  - compatibility: `ok=true`

## Contracts + Compatibility

- Existing run lifecycle event names were preserved (`RUN_CREATED`, `RUN_PREPARING`, `RUN_STARTED`, etc.).
- Existing v1 shapes remain compatible; additions are optional/additive (telemetry files, dashboard payload extensions, extra trainer option fields).
- Legacy baseline trainer remains intact as fallback (`model.type` omitted or `baseline_sgd`).

## Remaining Risks / Follow-ups

1. **Resume-safe checkpointing depth**
   - Current hardening provides atomic JSON checkpoint writes and progress-safe checkpoint capture.
   - Full optimizer/model-state resume for all trainer families is not yet universal.

2. **Telemetry hardware richness**
   - GPU metrics currently cover VRAM used/total (CUDA path). GPU utilization % would require NVML integration where available.

3. **Digest cadence semantics**
   - Current dedupe/cooldown prevents notification spam and persists state, but operators may want stricter “single final digest per batch-id” semantics in future.
