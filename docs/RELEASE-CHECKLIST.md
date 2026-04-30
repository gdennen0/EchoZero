# EchoZero Release Checklist

Status: reference
Last reviewed: 2026-04-30


Use this checklist for release-affecting work and milestone signoff.

This complements `docs/APP-DELIVERY-PLAN.md`; it does not replace it.

## Code And Contract Gates

- contract and unit lanes are green
- changed app-facing flows are proven through app-level lanes
- sync changes preserve main-only constraints and confirm/apply safety
- no widget-only workflow logic bypasses the application layer

## App And UX Gates

- `run_echozero.py` remains the canonical app path
- visible operator flows still work through the real app shell
- timeline changes preserve FEEL and do not introduce magic-number drift
- manual QA is performed for milestone or release-signoff work

## Packaging Gates

- build path succeeds
- packaged smoke path succeeds
- release artifact output is deterministic enough for inspection

## Evidence Bundle

Capture:

- commands run
- pass/fail outputs
- packaged smoke result
- manual QA notes when required

## Special Attention Areas

- main versus takes truth model
- stale propagation after upstream-main changes
- MA3 pull/push safety semantics
- sync receive path regressions
- release packaging regressions
