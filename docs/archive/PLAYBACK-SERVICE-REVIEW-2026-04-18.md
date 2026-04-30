# Playback Service Review

Status: historical
Last reviewed: 2026-04-30


Originally updated: 2026-04-18

This review is a point-in-time remediation checkpoint.
For current playback boundaries, use `docs/STATUS.md`.

This is the PB-34 review gate for the playback service boundary work.

## Scope Reviewed

- [echozero/application/playback/runtime.py](../../echozero/application/playback/runtime.py)
- [echozero/ui/qt/timeline/runtime_audio.py](../../echozero/ui/qt/timeline/runtime_audio.py)
- [echozero/ui/qt/app_shell.py](../../echozero/ui/qt/app_shell.py)
- [echozero/ui/qt/timeline/demo_app.py](../../echozero/ui/qt/timeline/demo_app.py)
- [tests/ui/test_runtime_audio.py](../../tests/ui/test_runtime_audio.py)
- [tests/ui/test_app_shell_runtime_flow.py](../../tests/ui/test_app_shell_runtime_flow.py)

## Findings

No major architectural regressions found in the reviewed slice.

## What Passed Review

- playback target resolution now lives in application playback runtime code, not widget convenience code
- selection fallback was not reintroduced
- UI still consumes narrow runtime methods and timing snapshots rather than raw backend internals
- app shell now receives backend session metadata through the runtime boundary
- compatibility surface remains thin: `echozero/ui/qt/timeline/runtime_audio.py` is now an import shim rather than the semantic owner

## Residual Risks

- the current runtime boundary is still presentation-backed, not full timeline-domain-backed; that is acceptable for the near-term shell but should stay explicit
- `snapshot_state()` derives active source metadata from the current presentation-backed target, so malformed or stale presentation inputs would still surface there
- PB-32 and PB-33 still require real hardware/manual validation for audible correctness and device-format behavior
- PB-54 is still open because the documented benchmark command did not resolve to a runnable benchmark in this environment

## Review Decision

PB-34 passes for the current remediation slice.

There is no blocking contract leakage or selection-coupled playback regression left in the reviewed code.
