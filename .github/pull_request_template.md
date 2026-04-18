## Why this change

- Problem:
- Why now:
- User-visible impact:

## Decision / principle mapping (required)

- Implements:
  - [ ] FP / decision ID(s) listed explicitly
- Contract touched:
  - [ ] Timeline truth model
  - [ ] Inspector contract
  - [ ] Sync boundary
  - [ ] FEEL/UI shell constants
  - [ ] Other: 

## What changed

- 

## Agent workflow

- Lead/orchestrator:
- Disposable agents used:
  - [ ] `0`
  - [ ] listed below
- Delegated ownership boundaries:
- Why delegation was used or skipped:

### Spawn proof

- `spawned`:
- `role`:
- `ownership`:
- `status`:
- `closeout`:

## Risk + rollback

- Risk:
- Rollback path:

## Verification

- [ ] Focused tests run and pasted below
- [ ] Real-data verification done (if visible timeline/sync behavior changed)
- [ ] Perf guardrail lane run (if hot paths changed)
- [ ] `appflow` listed below when timeline/UI/app-shell behavior changed
- [ ] `appflow-sync` and `appflow-protocol` listed below when sync behavior changed
- [ ] Explain why a lane was skipped if the touched surface normally requires it

### Test output

```text
paste output here
```

## Hygiene checks

- [ ] No generated/runtime outputs added to git
- [ ] `git status` clean before push
- [ ] Disposable agents/sessions used for this change were closed or explicitly justified as still open
