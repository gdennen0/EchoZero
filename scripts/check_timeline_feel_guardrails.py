#!/usr/bin/env python3
"""
Timeline FEEL guardrails: reject new hard-coded UI tuning literals in changed timeline files.
Exists so new timeline sizing and interaction drift moves into FEEL/style rather than widget code.
Connects AGENT-CONTEXT FEEL rules to a narrow, changed-file CI check.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SCOPES = ("echozero/ui/qt/timeline/", "echozero/ui/qt/app_shell_project_timeline.py")
EXCLUDED_SUFFIXES = (
    "demo_app.py",
    "test_harness.py",
    "style.py",
    "real_data_fixture.py",
    "drum_classifier_preview.py",
    "waveform_cache.py",
)
ALLOWED_LITERALS = {0, 0.0, 1, 1.0, -1, -1.0, 2, 2.0}
BASELINE_ALLOWLIST: dict[str, set[int]] = {
    "echozero/ui/qt/timeline/manual_pull.py": {45, 46, 47, 48, 49},
    "echozero/ui/qt/timeline/widget.py": {84, 135},
}


def _git_changed_files(repo_root: Path, *, base: str, head: str) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", "--diff-filter=ACMR", f"{base}..{head}"],
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
        if normalized.endswith(EXCLUDED_SUFFIXES):
            continue
        candidate = repo_root / normalized
        if candidate.exists():
            files.append(candidate)
    return files


def _is_numeric_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value not in ALLOWED_LITERALS
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
            return (-operand.value) not in ALLOWED_LITERALS
    return False


def _is_feel_or_style_symbol(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id.isupper() or node.id.endswith("_STYLE")
    if isinstance(node, ast.Attribute):
        return node.attr.isupper() or node.attr.endswith("_STYLE")
    return False


def _check_assignment(path: Path, node: ast.AST) -> list[str]:
    value = getattr(node, "value", None)
    if value is None:
        return []
    if _is_feel_or_style_symbol(value):
        return []
    if _is_numeric_constant(value):
        return [f"{path}:{node.lineno}: move UI tuning literal {ast.unparse(value)} into FEEL/style"]
    return []


def _check_call_defaults(path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    failures: list[str] = []
    for default in node.args.defaults:
        if _is_numeric_constant(default):
            failures.append(
                f"{path}:{node.lineno}: move numeric default {ast.unparse(default)} into FEEL/style or a named constant"
            )
    for default in node.args.kw_defaults:
        if default is not None and _is_numeric_constant(default):
            failures.append(
                f"{path}:{node.lineno}: move keyword numeric default {ast.unparse(default)} into FEEL/style or a named constant"
            )
    return failures


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    repo_root = Path(__file__).resolve().parents[1]
    relative_path = path.resolve().relative_to(repo_root).as_posix()
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(module):
        if getattr(node, "lineno", None) in BASELINE_ALLOWLIST.get(relative_path, set()):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            failures.extend(_check_assignment(path, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.extend(_check_call_defaults(path, node))
    return failures


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        print("usage: check_timeline_feel_guardrails.py <base> <head>")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    changed_files = _git_changed_files(repo_root, base=argv[0], head=argv[1])
    if not changed_files:
        print("Timeline FEEL guardrails passed: no guarded timeline files changed.")
        return 0

    failures: list[str] = []
    for path in changed_files:
        failures.extend(_check_file(path))

    if failures:
        print("Timeline FEEL guardrails failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Timeline FEEL guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
