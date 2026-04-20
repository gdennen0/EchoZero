#!/usr/bin/env python3
"""
Canonical launcher guardrails: keep run_echozero.py as the single desktop entrypoint.
Exists so packaging and smoke paths cannot silently drift to alternate launch surfaces.
Connects APP-DELIVERY-PLAN launcher rules to CI enforcement.
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_FILES = (
    "run_echozero.py",
    "scripts/build-test-release.ps1",
    "scripts/smoke-test-release.ps1",
)
FORBIDDEN_FILES = (
    "main.py",
    "main_qt.py",
)
REQUIRED_REFERENCES = {
    "scripts/build-test-release.ps1": ("run_echozero.py",),
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    repo_root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]

    failures: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (repo_root / relative_path).exists():
            failures.append(f"missing required launcher file: {relative_path}")

    for relative_path in FORBIDDEN_FILES:
        if (repo_root / relative_path).exists():
            failures.append(f"forbidden legacy launcher reintroduced: {relative_path}")

    for relative_path, required_terms in REQUIRED_REFERENCES.items():
        content = (repo_root / relative_path).read_text(encoding="utf-8")
        for term in required_terms:
            if term not in content:
                failures.append(f"{relative_path}: missing required launcher reference '{term}'")

    if failures:
        print("Canonical launcher check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Canonical launcher check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
