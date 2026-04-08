#!/usr/bin/env python
"""Repository hygiene guardrails.

Fails CI when generated/runtime output is accidentally tracked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "artifacts/",
    "foundry/runs/",
    "foundry/state/",
    "foundry/tracking/batches/",
    "foundry/tracking/exports/",
    "foundry/tracking/model_cards/",
    "foundry/tracking/snapshots/",
)

# Keep curated top-level docs in tracking area; block noisy machine outputs only.
FORBIDDEN_EXACT = {
    "foundry/tracking/training_index.json",
    "foundry/tracking/notification_state.json",
}


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tracked = _git_ls_files(repo_root)

    offenders: list[str] = []
    for path in tracked:
        normalized = path.replace("\\", "/")
        if normalized in FORBIDDEN_EXACT:
            offenders.append(normalized)
            continue
        if normalized.startswith(FORBIDDEN_PREFIXES):
            offenders.append(normalized)

    if offenders:
        print("Repo hygiene check failed: generated/runtime outputs are tracked:")
        for item in offenders:
            print(f"  - {item}")
        print("\nRemove from Git and keep generated outputs local (see .gitignore).")
        return 1

    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
