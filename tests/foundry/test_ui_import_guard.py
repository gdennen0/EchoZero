from __future__ import annotations

import ast
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[2] / "echozero" / "foundry" / "ui"
FORBIDDEN_IMPORT_PREFIX = "echozero.foundry.persistence"


def _is_forbidden_module(module: str) -> bool:
    return module == FORBIDDEN_IMPORT_PREFIX or module.startswith(f"{FORBIDDEN_IMPORT_PREFIX}.")


def test_foundry_ui_does_not_import_persistence_repositories_directly() -> None:
    violations: list[str] = []

    for path in sorted(UI_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and _is_forbidden_module(module):
                    violations.append(f"{path.name}:{node.lineno} imports from {module}")
                if node.level > 0 and module.startswith("persistence"):
                    violations.append(
                        f"{path.name}:{node.lineno} uses relative import from persistence ({module})"
                    )

    assert not violations, (
        "Foundry UI must stay app-layer only and cannot directly import persistence repositories. "
        f"Violations: {violations}"
    )
