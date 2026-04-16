# EchoZero Testing Guide

This file is the execution map for proving work in EchoZero.

Choose the smallest lane that proves the behavior, then expand only if the
change touches broader product boundaries.

## Baseline Setup

Create and activate the local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Core Commands

Full pytest baseline:

```bash
pytest tests/ -x --timeout=30
```

Canonical app-flow lanes:

```bash
python -m echozero.testing.run --lane appflow
python -m echozero.testing.run --lane appflow-sync
python -m echozero.testing.run --lane appflow-osc
python -m echozero.testing.run --lane appflow-protocol
python -m echozero.testing.run --lane appflow-all
python -m echozero.testing.run --lane gui-lane-b
```

Repo hygiene:

```bash
python scripts/check_repo_hygiene.py
```

Perf guardrail hotspot:

```bash
pytest tests/benchmarks/benchmark_timeline_phase3.py -q
```

## Lane Selection

### Pure contract or domain change

Use targeted pytest modules first.

Examples:

- domain types or graph invariants
- application models and intent translation
- repository or mapper behavior

### Timeline behavior change

Run:

1. targeted pytest slice
2. relevant app-flow lane
3. perf guardrail if the change affects hot timeline paths

### Sync or MA3 behavior change

Run:

1. targeted sync tests
2. `python -m echozero.testing.run --lane appflow-sync`
3. `python -m echozero.testing.run --lane appflow-protocol`
4. OSC or simulator lanes if transport or receive path changed

### UI shell or operator workflow change

Run:

1. targeted pytest slice
2. `python -m echozero.testing.run --lane appflow`
3. `python -m echozero.testing.run --lane gui-lane-b` when visible timeline behavior changed

### Release-affecting change

Also run the packaging path documented in `docs/APP-DELIVERY-PLAN.md`.

## Proof Standards

- Demo-only proof is not sufficient.
- App-facing changes are not done until proven through the app path.
- Main-only sync boundaries must stay provable.
- When a test cannot be run, state why and name the next best proof lane.

## Reporting Contract

Every meaningful verification report should include:

- commands run
- files or feature area covered
- pass/fail result
- remaining risks or untested surfaces
