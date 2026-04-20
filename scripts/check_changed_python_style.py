#!/usr/bin/env python3
"""
Changed-file style guardrails: enforce module headers, public docstrings, and file size.
Exists to turn STYLE.md expectations into automation without blocking untouched legacy files.
Connects PR/push diffs to the repo's maintainability rules for new changes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

MAX_LINES = 500
SCOPES = ("echozero/", "tests/", "scripts/", "run_")


class PublicSymbolDocState(NamedTuple):
    """Docstring state for one public top-level symbol."""

    line: int
    has_docstring: bool


def _git_changed_files(repo_root: Path, *, base: str, head: str) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base}..{head}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    files: list[Path] = []
    for raw_path in result.stdout.splitlines():
        normalized = raw_path.strip().replace("\\", "/")
        if not normalized.endswith(".py"):
            continue
        if not normalized.startswith(SCOPES):
            continue
        candidate = repo_root / normalized
        if candidate.exists():
            files.append(candidate)
    return files


def _git_show_file(repo_root: Path, *, revision: str, relative_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{revision}:{relative_path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _is_test_file(path: Path) -> bool:
    return "tests" in path.parts


def _has_three_line_header(module: ast.Module) -> bool:
    docstring = ast.get_docstring(module, clean=False)
    if docstring is None:
        return False
    lines = [line.strip() for line in docstring.splitlines() if line.strip()]
    return len(lines) >= 3


def _public_members_without_docstrings(module: ast.Module) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node) is None:
                failures.append((node.name, node.lineno))
    return failures


def _public_member_doc_map(module: ast.Module) -> dict[str, PublicSymbolDocState]:
    members: dict[str, PublicSymbolDocState] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            members[node.name] = PublicSymbolDocState(
                line=node.lineno,
                has_docstring=ast.get_docstring(node) is not None,
            )
    return members


def _check_file(path: Path, *, prior_source: str | None) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())

    module = ast.parse(source, filename=str(path))
    prior_module = (
        ast.parse(prior_source, filename=str(path)) if prior_source is not None else None
    )
    prior_line_count = len(prior_source.splitlines()) if prior_source is not None else None

    if line_count > MAX_LINES and prior_line_count is None:
        errors.append(f"{path}: exceeds hard limit with {line_count} lines (max {MAX_LINES})")

    if not _has_three_line_header(module) and (
        prior_module is None or _has_three_line_header(prior_module)
    ):
        errors.append(f"{path}: missing 3-line module header docstring")

    if not _is_test_file(path):
        current_members = _public_member_doc_map(module)
        prior_members = _public_member_doc_map(prior_module) if prior_module is not None else {}
        for name, state in current_members.items():
            prior_state = prior_members.get(name)
            if state.has_docstring:
                continue
            if prior_state is not None and not prior_state.has_docstring:
                continue
            errors.append(f"{path}:{state.line}: public symbol '{name}' is missing a docstring")
    return errors


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        print("usage: check_changed_python_style.py <base> <head>")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    changed_files = _git_changed_files(repo_root, base=argv[0], head=argv[1])
    if not changed_files:
        print("Changed-file style check passed: no Python files in guarded scopes changed.")
        return 0

    failures: list[str] = []
    for path in changed_files:
        relative_path = path.relative_to(repo_root)
        prior_source = _git_show_file(repo_root, revision=argv[0], relative_path=relative_path)
        failures.extend(_check_file(path, prior_source=prior_source))

    if failures:
        print("Changed-file style check failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Changed-file style check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
