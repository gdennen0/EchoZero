#!/usr/bin/env python3
"""
Lane ownership guard: validate changed paths against the current branch's lane.
Exists so parallel EZ and Foundry worktrees do not drift into the same file clusters.
Connects docs/DEV-LANES.md ownership rules to a cross-platform local check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

LANE_RULES = {
    "foundry": ("echozero/foundry/", "tests/foundry/"),
    "ez": ("echozero/application/", "echozero/ui/", "tests/application/", "tests/ui/"),
}
SHARED_RULES = (
    "echozero/inference_eval/",
    "tests/inference_eval/",
    "tests/processors/test_pytorch_audio_classify_preflight.py",
)


def _git_current_branch(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_changed_files(repo_root: Path, *, base_ref: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}...HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _infer_lane(branch_name: str) -> str | None:
    if branch_name.startswith("lane/foundry"):
        return "foundry"
    if branch_name.startswith("lane/ez"):
        return "ez"
    if branch_name.startswith("lane/integration"):
        return "integration"
    return None


def _git_action_branch_name() -> str | None:
    head_ref = os.environ.get("GITHUB_HEAD_REF")
    if head_ref:
        return head_ref.strip() or None
    ref_name = os.environ.get("GITHUB_REF_NAME")
    if ref_name:
        return ref_name.strip() or None
    return None


def _is_opposing_lane_path(path: str, *, lane: str) -> bool:
    opposing_lane = "foundry" if lane == "ez" else "ez"
    return path.startswith(LANE_RULES[opposing_lane])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate changed paths against lane ownership.")
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--lane", choices=["ez", "foundry", "integration"])
    parser.add_argument("--allow-shared", action="store_true")
    parsed = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    branch_name = _git_action_branch_name() or _git_current_branch(repo_root)
    lane = parsed.lane or _infer_lane(branch_name)
    if lane is None:
        print(f"Lane ownership check skipped: branch '{branch_name}' is not a lane branch.")
        return 0

    if lane == "integration":
        print("Lane ownership check passed: integration branches may touch shared zones.")
        return 0

    changed = _git_changed_files(repo_root, base_ref=parsed.base_ref)
    offenders: list[str] = []

    for path in changed:
        if path.startswith(LANE_RULES[lane]):
            continue
        if path.startswith(SHARED_RULES):
            if parsed.allow_shared:
                continue
            offenders.append(path)
            continue
        if _is_opposing_lane_path(path, lane=lane):
            offenders.append(path)

    if offenders:
        print(f"Lane ownership check failed for lane '{lane}':")
        for offender in offenders:
            print(f"  - {offender}")
        print("Use an integration branch or pass --allow-shared for shared-zone work.")
        return 1

    print(f"Lane ownership check passed for lane '{lane}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
