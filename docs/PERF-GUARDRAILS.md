# Performance Guardrails (Phase 3)

Status: reference
Last reviewed: 2026-04-30


This repo includes a deterministic timeline perf guardrail benchmark:

- Script: `tests/benchmarks/benchmark_timeline_phase3.py`
- Thresholds: `tests/benchmarks/timeline_phase3_thresholds.json`
- CI output artifact: `artifacts/perf/timeline_phase3.json`

## What it measures

1. **Cached timeline assembly latency**
   - Transport-only updates should reuse assembled layers quickly.
   - Metrics: p50/p95/max milliseconds.

2. **Event lane paint latency (dense timeline)**
   - Paint cost under high event density with viewport culling active.
   - Metrics: p50/p95/max milliseconds.

## Local run

```bash
set QT_QPA_PLATFORM=offscreen
set PYTHONPATH=C:\Users\griff\EchoZero
.\.venv\Scripts\python.exe tests\benchmarks\benchmark_timeline_phase3.py --strict --json-out artifacts\perf\timeline_phase3.json
```

If `--strict` is passed, the script exits non-zero when thresholds are exceeded.

## Tuning thresholds

Edit `tests/benchmarks/timeline_phase3_thresholds.json`.

Keep thresholds realistic and stable across CI runners; prefer adjusting based on sustained measurements, not one-off spikes.
