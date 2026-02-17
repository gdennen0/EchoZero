#!/usr/bin/env python3
"""
Quality assurance system for AI agent contributions.

This script provides automated quality checks to ensure AI-generated code
maintains EchoZero's standards and core values.
"""

import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess


class QualityChecker:
    """Automated quality checks for AI agent contributions."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.core_values_path = self.project_root / "AgentAssets" / "core" / "CORE_VALUES.md"

    def check_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Run quality checks on specified files."""
        results = {}

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue

            issues = []
            issues.extend(self._check_core_values_violations(path))
            issues.extend(self._check_architecture_consistency(path))
            issues.extend(self._check_documentation_requirements(path))
            issues.extend(self._check_code_quality(path))

            if issues:
                results[str(path)] = issues

        return results

    def _check_core_values_violations(self, file_path: Path) -> List[str]:
        """Check for violations of 'best part is no part' principle."""
        issues = []

        if file_path.suffix != '.py':
            return issues

        content = file_path.read_text()

        # Check for unnecessary abstractions
        if self._has_unnecessary_abstraction(content):
            issues.append("Potential unnecessary abstraction - review if this complexity is needed")

        # Check for new dependencies
        if self._introduces_new_dependency(content):
            issues.append("New dependency introduced - ensure it follows 'best part is no part'")

        # Check for over-engineering
        if self._is_over_engineered(content):
            issues.append("Potential over-engineering - consider simpler solution")

        return issues

    def _has_unnecessary_abstraction(self, content: str) -> bool:
        """Detect potentially unnecessary abstractions."""
        # Look for interfaces with single implementation
        interface_pattern = r'class \w+Interface|class \w+ABC|class \w+Abstract'
        implementations = re.findall(r'class \w+.*:', content)

        # Count implementations of interfaces
        interfaces = re.findall(interface_pattern, content)
        if len(interfaces) > len(implementations) * 2:  # Rough heuristic
            return True

        # Look for factory patterns for single types
        if 'Factory' in content and len(re.findall(r'class \w+Factory', content)) > 1:
            return True

        return False

    def _introduces_new_dependency(self, content: str) -> bool:
        """Check if new external dependencies are introduced."""
        import_lines = re.findall(r'^(?:import|from) \w+', content, re.MULTILINE)

        # Known allowed imports (expand as needed)
        allowed_imports = {
            'pathlib', 'os', 'sys', 'typing', 'dataclasses', 'abc',
            'numpy', 'torch', 'librosa', 'soundfile', 'PyQt6'
        }

        for line in import_lines:
            module = line.split()[1].split('.')[0]
            if module not in allowed_imports:
                return True

        return False

    def _is_over_engineered(self, content: str) -> bool:
        """Detect over-engineering patterns."""
        # Too many classes in one file
        if len(re.findall(r'^class \w+', content, re.MULTILINE)) > 5:
            return True

        # Complex inheritance hierarchies
        if content.count('):') > 3:  # Multiple inheritance or deep hierarchies
            return True

        # Too many decorators (potential over-abstraction)
        if len(re.findall(r'@\w+', content)) > 5:
            return True

        return False

    def _check_architecture_consistency(self, file_path: Path) -> List[str]:
        """Check architectural consistency."""
        issues = []

        content = file_path.read_text()

        # UI files shouldn't access domain/infrastructure directly
        if 'ui/qt_gui' in str(file_path):
            if any(layer in content for layer in ['domain.', 'infrastructure.']):
                issues.append("UI layer accessing domain/infrastructure directly - use ApplicationFacade")

        # Domain shouldn't depend on outer layers
        if 'domain' in str(file_path):
            if any(layer in content for layer in ['application.', 'infrastructure.', 'ui.']):
                issues.append("Domain layer depending on outer layers - violates layered architecture")

        return issues

    def _check_documentation_requirements(self, file_path: Path) -> List[str]:
        """Check documentation requirements."""
        issues = []

        if file_path.suffix != '.py':
            return issues

        content = file_path.read_text()

        # Check for docstrings on classes and functions
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if not ast.get_docstring(node):
                    issues.append(f"Missing docstring for {node.name}")

        # Check for complex functions without comments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 10 and not any(isinstance(n, ast.Expr) and
                                                  isinstance(n.value, ast.Str)
                                                  for n in node.body[:3]):
                    issues.append(f"Complex function {node.name} needs documentation")

        return issues

    def _check_code_quality(self, file_path: Path) -> List[str]:
        """General code quality checks."""
        issues = []

        if file_path.suffix != '.py':
            return issues

        content = file_path.read_text()

        # Check line length (rough heuristic)
        long_lines = [i+1 for i, line in enumerate(content.split('\n'))
                     if len(line) > 100]
        if long_lines:
            issues.append(f"Long lines found at: {long_lines[:5]}")

        # Check for TODO comments (should be addressed)
        todo_lines = [i+1 for i, line in enumerate(content.split('\n'))
                     if 'TODO' in line.upper()]
        if todo_lines:
            issues.append(f"TODO comments found at lines: {todo_lines}")

        # Check for print statements in production code
        if 'print(' in content and 'test' not in str(file_path):
            issues.append("Print statements found - use logging instead")

        return issues


def main():
    """Run quality checks from command line."""
    if len(sys.argv) < 2:
        print("Usage: python quality_checks.py <file1> [file2] ...")
        sys.exit(1)

    checker = QualityChecker()
    results = checker.check_files(sys.argv[1:])

    if not results:
        print("✅ All quality checks passed!")
        return 0

    print("❌ Quality issues found:")
    for file_path, issues in results.items():
        print(f"\n{file_path}:")
        for issue in issues:
            print(f"  - {issue}")

    return 1


if __name__ == '__main__':
    sys.exit(main())

