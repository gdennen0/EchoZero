#!/usr/bin/env python3
"""
PR proof guardrails: require the right verification notes for the change surface.
Exists so timeline, UI, and sync changes carry explicit proof lanes in the PR body.
Connects changed files to the verification contract in docs/TESTING.md and the PR template.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SYNC_PREFIXES = ("echozero/application/sync/", "echozero/infrastructure/sync/", "tests/application/")
TIMELINE_PREFIXES = ("echozero/application/timeline/", "echozero/ui/", "tests/ui/", "tests/gui/", "tests/testing/")


def _git_changed_files(repo_root: Path, *, base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", "--diff-filter=ACMR", f"{base}..{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _require(body: str, required_terms: list[str], *, label: str) -> list[str]:
    body_lower = body.lower()
    missing = [term for term in required_terms if term.lower() not in body_lower]
    if not missing:
        return []
    return [f"{label}: missing {', '.join(missing)} in PR body"]


def main() -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_name != "pull_request" or not event_path:
        print("PR requirements check skipped: not running in pull_request context.")
        return 0

    payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pr = payload["pull_request"]
    body = pr.get("body") or ""
    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]

    repo_root = Path(__file__).resolve().parents[1]
    changed = _git_changed_files(repo_root, base=base_sha, head=head_sha)

    failures: list[str] = []
    if any(path.startswith(SYNC_PREFIXES) for path in changed):
        failures.extend(
            _require(body, ["appflow-sync", "appflow-protocol"], label="Sync change verification")
        )
    if any(path.startswith(TIMELINE_PREFIXES) for path in changed):
        failures.extend(_require(body, ["appflow"], label="Timeline/UI change verification"))
    if any("tests/benchmarks/" in path for path in changed):
        failures.extend(_require(body, ["perf"], label="Perf change verification"))

    if failures:
        print("PR requirements check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PR requirements check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
