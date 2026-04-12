# Dev Lanes: Foundry + EZ Parallel Workflow

Use two isolated git worktrees so Foundry and EZ can move in parallel without stepping on each other.

## 1) Worktree setup

```powershell
git -C C:\Users\griff\EchoZero worktree add -b lane/foundry C:\Users\griff\EchoZero-foundry main
git -C C:\Users\griff\EchoZero worktree add -b lane/ez C:\Users\griff\EchoZero-ez main
```

Current lane directories:
- Foundry lane: `C:\Users\griff\EchoZero-foundry`
- EZ lane: `C:\Users\griff\EchoZero-ez`

## 2) Ownership rules

### Foundry-owned
- `echozero/foundry/**`
- `tests/foundry/**`

### EZ-owned
- `echozero/application/**`
- `echozero/ui/**`
- `tests/application/**`
- `tests/ui/**`

### Shared zone (requires integration gate)
- `echozero/inference_eval/**`
- `tests/inference_eval/**`
- `tests/processors/test_pytorch_audio_classify_preflight.py`

## 3) Integration gate for shared zone

If a lane touches shared-zone paths, merge that lane into a temporary integration branch first and run both lane gates before merging to `main`.

Suggested integration flow:

```powershell
git switch -c lane/integration main
# cherry-pick or merge lane/foundry and lane/ez work
# run gates, then merge to main
```

## 4) Lane guard script

Use `scripts/validate-lane-ownership.ps1` before commit/push.

Examples:

```powershell
# Validate staged files in foundry lane
powershell -File scripts/validate-lane-ownership.ps1 -Lane foundry -Mode staged

# Validate current uncommitted files in ez lane
powershell -File scripts/validate-lane-ownership.ps1 -Lane ez -Mode worktree

# Validate commits vs origin/main (default)
powershell -File scripts/validate-lane-ownership.ps1 -Lane foundry -Mode range -BaseRef origin/main

# Allow shared-zone edits (integration only)
powershell -File scripts/validate-lane-ownership.ps1 -Lane foundry -Mode range -AllowShared
```

## 5) Required gates

### Foundry lane
```powershell
python -m pytest tests/foundry -q
python -m pytest tests/processors/test_pytorch_audio_classify_preflight.py tests/inference_eval/test_runtime_preflight_compatibility.py tests/inference_eval/test_validation_core.py tests/foundry/test_artifact_shared_fingerprint.py tests/foundry/test_train_artifact_runtime_parity.py -q
```

### EZ lane
Run EZ app/UI gates for the active slice.

## 6) Reporting contract per update

Each lane update must include:
- commit hash
- files changed
- tests passed
- whether shared zone was touched
