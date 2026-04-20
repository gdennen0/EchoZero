#!/usr/bin/env python3
"""
Architecture boundary guardrails: keep core/application layers free of Qt and UI imports.
Exists so engine and application contracts stay independent from presentation implementation.
Connects AGENT-CONTEXT boundary rules to a fast CI check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

GUARDED_PREFIXES = (
    "echozero/domain/",
    "echozero/editor/",
    "echozero/persistence/",
    "echozero/services/",
    "echozero/application/",
    "echozero/infrastructure/",
)
FORBIDDEN_IMPORT_PREFIXES = ("PyQt6", "echozero.ui")


def _iter_guarded_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for prefix in GUARDED_PREFIXES:
        root = repo_root / prefix
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


def _import_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        else:
            continue

        for name in names:
            for prefix in FORBIDDEN_IMPORT_PREFIXES:
                if _import_matches(name, prefix):
                    failures.append(f"{path}:{node.lineno}: forbidden import '{name}' in guarded layer")
    return failures


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    repo_root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]

    failures: list[str] = []
    for path in _iter_guarded_files(repo_root):
        failures.extend(_check_file(path))

    if failures:
        print("Architecture boundary check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Architecture boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
