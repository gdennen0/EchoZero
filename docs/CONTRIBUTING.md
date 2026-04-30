# Contributing (Git Publish Guardrails)

Status: reference
Last reviewed: 2026-04-30


This repo uses a **branch + PR publish workflow**. Treat `origin/main` as the source of truth.

## Publish flow (avoid local-main/origin-main drift)

1. **Fetch origin first**
   ```bash
   git fetch origin
   ```

2. **Inspect divergence: local `main` vs `origin/main`**
   ```bash
   git switch main
   git rev-list --left-right --count main...origin/main
   git log --oneline --decorate --graph --max-count=12 main origin/main
   ```
   - Output is `A B` from `rev-list`:
     - `A > 0`: local `main` is ahead of `origin/main`
     - `B > 0`: local `main` is behind `origin/main`
   - If either side is non-zero, reconcile before publishing.

3. **Create/update a feature branch from `origin/main`**
   ```bash
   git switch -c <ticket-or-topic> origin/main
   # make changes
   git add <explicit-files>
   git commit -m "<clear message>"
   git push -u origin <ticket-or-topic>
   ```

4. **Open PR: `<ticket-or-topic>` -> `main`**
   - Merge through PR (no direct push from local `main`).

## Workspace bootstrap

Use the scripted bootstrap instead of ad hoc setup:

```bash
python3 scripts/dev_bootstrap.py
```

Optional stacks:

```bash
python3 scripts/dev_bootstrap.py --extras ml packaging
```

Recommended local hooks:

```bash
.venv/bin/pre-commit install
```

Use `.work/` for temporary notes, logs, generated diffs, and throwaway artifacts.
Use [docs/agent-task-template.md](agent-task-template.md) when handing work to
another agent or a separate worktree.

## Pre-push checklist (exact commands)

Run these before any branch push:

```bash
git fetch origin
git status --short
git rev-list --left-right --count main...origin/main
git log --oneline --decorate --graph --max-count=8 HEAD origin/main
git diff --name-only --cached
```

Quick pass criteria:
- Working tree only includes intended files.
- `main`/`origin/main` divergence is understood.
- Staged file list matches ticket scope.
- You are pushing a topic branch, not `main`.
