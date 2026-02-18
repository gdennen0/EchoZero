#!/usr/bin/env python3
"""
Cursor IDE Integration Hooks for EchoZero.

This module provides hooks that integrate with Cursor IDE's file operations,
automatically maintaining AgentAssets synchronization and quality standards.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import time


class CursorIDEHooks:
    """Hooks for Cursor IDE integration."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.agent_assets_root = self.project_root / "AgentAssets"
        self.last_sync_time = 0
        self.sync_cooldown = 5  # seconds between syncs

    def on_file_save(self, file_path: str) -> Dict[str, Any]:
        """
        Called when a file is saved in Cursor IDE.

        Performs quality checks and synchronization as needed.
        """
        file_path = Path(file_path)

        # Skip if not in project
        if not file_path.is_relative_to(self.project_root):
            return {"status": "skipped", "reason": "outside project"}

        result = {
            "status": "success",
            "checks_run": [],
            "issues_found": [],
            "sync_performed": False
        }

        # Run quality checks on Python and Markdown files
        if file_path.suffix in ['.py', '.md']:
            quality_result = self._run_quality_checks(file_path)
            result["checks_run"].append("quality")
            result["issues_found"].extend(quality_result.get("issues", []))

            if quality_result.get("status") == "failed":
                result["status"] = "warning"

        # Trigger sync for critical files
        if self._should_trigger_sync(file_path):
            if time.time() - self.last_sync_time > self.sync_cooldown:
                sync_result = self._run_incremental_sync()
                result["sync_performed"] = sync_result.get("status") == "success"
                self.last_sync_time = time.time()

        return result

    def on_project_open(self) -> Dict[str, Any]:
        """
        Called when the project is opened in Cursor IDE.

        Performs initial synchronization and validation.
        """
        result = {
            "status": "success",
            "sync_performed": False,
            "validation_passed": True,
            "issues_found": []
        }

        # Run full synchronization
        try:
            sync_result = self._run_full_sync()
            result["sync_performed"] = sync_result.get("status") == "success"
        except Exception as e:
            result["status"] = "error"
            result["issues_found"].append(f"Sync failed: {e}")

        # Validate AgentAssets integrity
        try:
            validation_result = self._validate_agent_assets()
            result["validation_passed"] = validation_result.get("valid", False)
            result["issues_found"].extend(validation_result.get("issues", []))
        except Exception as e:
            result["status"] = "error"
            result["issues_found"].append(f"Validation failed: {e}")

        return result

    def on_file_create(self, file_path: str) -> Dict[str, Any]:
        """
        Called when a new file is created in Cursor IDE.

        Sets up templates and performs initial validation.
        """
        file_path = Path(file_path)

        result = {
            "status": "success",
            "template_applied": False,
            "checks_run": [],
            "context_available": False
        }

        # Apply templates for new files
        if file_path.suffix == '.py':
            template_result = self._apply_python_template(file_path)
            result["template_applied"] = template_result.get("applied", False)

        # Run initial quality checks
        if file_path.suffix in ['.py', '.md']:
            quality_result = self._run_quality_checks(file_path)
            result["checks_run"].append("quality")
            if quality_result.get("status") == "failed":
                result["status"] = "warning"

        # Initialize context for new file
        try:
            from context_injector import get_context_injector
            injector = get_context_injector()

            # Create a temporary tab ID for context generation
            temp_tab_id = f"new_file_{hash(file_path)}"
            context = injector.get_context_for_tab({
                "tab_id": temp_tab_id,
                "file_path": str(file_path),
                "request_type": "file_creation",
                "agent_id": "cursor_system"
            })

            if "error" not in context:
                result["context_available"] = True
                result["initial_context"] = context

        except Exception as e:
            result["context_error"] = str(e)

        return result

    def on_commit_attempt(self, files: List[str]) -> Dict[str, Any]:
        """
        Called before a commit attempt in Cursor IDE.

        Runs comprehensive validation before allowing commit.
        """
        result = {
            "status": "approved",
            "checks_passed": True,
            "blocking_issues": [],
            "warnings": []
        }

        # Run quality checks on all modified files
        for file_path in files:
            file_path = Path(file_path)
            if file_path.suffix in ['.py', '.md'] and file_path.exists():
                quality_result = self._run_quality_checks(file_path)

                if quality_result.get("status") == "failed":
                    result["checks_passed"] = False
                    result["blocking_issues"].extend(quality_result.get("issues", []))

                warnings = quality_result.get("warnings", [])
                result["warnings"].extend(warnings)

        # Run council decision validation if applicable
        council_files = [f for f in files if 'council_decision.json' in f]
        for council_file in council_files:
            validation_result = self._validate_council_decision(council_file)
            if not validation_result.get("valid", False):
                result["checks_passed"] = False
                result["blocking_issues"].extend(validation_result.get("errors", []))

        # Block commit if there are critical issues
        if result["blocking_issues"]:
            result["status"] = "blocked"

        return result

    def _run_quality_checks(self, file_path: Path) -> Dict[str, Any]:
        """Run quality checks on a file."""
        try:
            script_path = self.agent_assets_root / "scripts" / "quality_checks.py"
            result = subprocess.run([
                sys.executable, str(script_path), str(file_path)
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                return {"status": "passed", "issues": [], "warnings": []}
            else:
                # Parse output for issues
                issues = []
                warnings = []
                for line in result.stderr.split('\n'):
                    if line.strip():
                        if '❌' in line or 'ERROR' in line.upper():
                            issues.append(line.strip())
                        elif '⚠️' in line or 'WARNING' in line.upper():
                            warnings.append(line.strip())

                return {
                    "status": "failed" if issues else "warning",
                    "issues": issues,
                    "warnings": warnings
                }

        except Exception as e:
            return {"status": "error", "issues": [f"Quality check failed: {e}"], "warnings": []}

    def _run_incremental_sync(self) -> Dict[str, Any]:
        """Run incremental AgentAssets synchronization."""
        try:
            script_path = self.agent_assets_root / "scripts" / "auto_sync.py"
            result = subprocess.run([
                sys.executable, str(script_path), "sync"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            return {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "error": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Sync timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_full_sync(self) -> Dict[str, Any]:
        """Run full AgentAssets synchronization."""
        return self._run_incremental_sync()  # Same script, different context

    def _validate_agent_assets(self) -> Dict[str, Any]:
        """Validate AgentAssets integrity."""
        issues = []

        # Check if critical files exist
        critical_files = [
            "AgentAssets/core/CORE_VALUES.md",
            "AgentAssets/core/CURRENT_STATE.md",
            "AgentAssets/MODULE_INDEX.md",
            "AgentAssets/scripts/quality_checks.py"
        ]

        for file_path in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing critical file: {file_path}")

        # Check if scripts are executable
        script_dir = self.agent_assets_root / "scripts"
        if script_dir.exists():
            for script in script_dir.glob("*.py"):
                if not os.access(script, os.X_OK):
                    # Try to make executable
                    try:
                        script.chmod(script.stat().st_mode | 0o111)
                    except:
                        issues.append(f"Script not executable: {script.name}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _validate_council_decision(self, file_path: str) -> Dict[str, Any]:
        """Validate a council decision file."""
        try:
            script_path = self.agent_assets_root / "scripts" / "validate_council_decision.py"
            result = subprocess.run([
                sys.executable, str(script_path), file_path
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                return {"valid": True, "errors": []}
            else:
                errors = [line.strip() for line in result.stderr.split('\n') if line.strip()]
                return {"valid": False, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Validation failed: {e}"]}

    def _should_trigger_sync(self, file_path: Path) -> bool:
        """Determine if a file change should trigger synchronization."""
        # Trigger sync for critical files
        critical_patterns = [
            "src/application/blocks/registry.py",
            "src/domain/entities/block.py",
            "src/application/commands/",
            "AgentAssets/modules/",
            "docs/encyclopedia/"
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in critical_patterns)

    def _apply_python_template(self, file_path: Path) -> Dict[str, Any]:
        """Apply Python file template."""
        # Check if file is empty or minimal
        if file_path.exists() and file_path.stat().st_size > 100:
            return {"applied": False, "reason": "file not empty"}

        # Generate basic Python template
        template = f'''"""
{file_path.stem.replace("_", " ").title()}

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

# TODO: Add imports
# TODO: Add class/function definitions
# TODO: Add implementation

if __name__ == "__main__":
    pass
'''

        try:
            file_path.write_text(template)
            return {"applied": True}
        except Exception as e:
            return {"applied": False, "error": str(e)}


# Global hook instance
_cursor_hooks = CursorIDEHooks()


def on_file_save(file_path: str) -> Dict[str, Any]:
    """Cursor IDE hook: called on file save."""
    return _cursor_hooks.on_file_save(file_path)


def on_project_open() -> Dict[str, Any]:
    """Cursor IDE hook: called on project open."""
    return _cursor_hooks.on_project_open()


def on_file_create(file_path: str) -> Dict[str, Any]:
    """Cursor IDE hook: called on file creation."""
    return _cursor_hooks.on_file_create(file_path)


def on_commit_attempt(files: List[str]) -> Dict[str, Any]:
    """Cursor IDE hook: called before commit."""
    return _cursor_hooks.on_commit_attempt(files)


def on_tab_opened(tab_id: str, file_path: str, cursor_position: Dict[str, int] = None, agent_id: str = "cursor_user") -> Dict[str, Any]:
    """Cursor IDE hook: called when a tab is opened or switched to."""
    try:
        from context_injector import get_context_injector
        injector = get_context_injector()

        context = injector.get_context_for_tab({
            "tab_id": tab_id,
            "file_path": file_path,
            "cursor_position": cursor_position,
            "request_type": "tab_open",
            "agent_id": agent_id
        })

        return {
            "status": "success",
            "context_provided": True,
            "context": context,
            "relevance_score": context.get("relevance_score", 0)
        }

    except Exception as e:
        return {
            "status": "error",
            "context_provided": False,
            "error": str(e)
        }


def on_cursor_moved(tab_id: str, file_path: str, cursor_position: Dict[str, int], agent_id: str = "cursor_user") -> Dict[str, Any]:
    """Cursor IDE hook: called when cursor position changes significantly."""
    try:
        from context_injector import get_context_injector
        injector = get_context_injector()

        # Only provide context if position changed significantly (every 5 lines or so)
        # This prevents excessive context generation

        context = injector.get_context_for_tab({
            "tab_id": tab_id,
            "file_path": file_path,
            "cursor_position": cursor_position,
            "request_type": "cursor_move",
            "agent_id": agent_id
        })

        return {
            "status": "success",
            "context_updated": True,
            "context": context
        }

    except Exception as e:
        return {
            "status": "error",
            "context_updated": False,
            "error": str(e)
        }


def on_selection_changed(tab_id: str, selection: Dict[str, Any], agent_id: str = "cursor_user") -> Dict[str, Any]:
    """Cursor IDE hook: called when text selection changes."""
    try:
        from context_injector import get_context_injector
        injector = get_context_injector()

        context = injector.get_selection_context(tab_id, selection, agent_id)

        return {
            "status": "success",
            "selection_context": context
        }

    except Exception as e:
        return {
            "status": "error",
            "selection_context": None,
            "error": str(e)
        }


def on_text_input(tab_id: str, current_input: str, agent_id: str = "cursor_user") -> Dict[str, Any]:
    """Cursor IDE hook: called during text input for predictive context."""
    try:
        from context_injector import get_context_injector
        injector = get_context_injector()

        context = injector.get_predictive_context(tab_id, current_input, agent_id)

        return {
            "status": "success",
            "predictive_context": context
        }

    except Exception as e:
        return {
            "status": "error",
            "predictive_context": None,
            "error": str(e)
        }


# For testing/development
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hooks.py <command> [args...]")
        print("Commands: save <file>, open, create <file>, commit <files...>, context <file> [line]")
        sys.exit(1)

    hooks = CursorIDEHooks()
    command = sys.argv[1]

    if command == "save" and len(sys.argv) > 2:
        result = hooks.on_file_save(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif command == "open":
        result = hooks.on_project_open()
        print(json.dumps(result, indent=2))

    elif command == "create" and len(sys.argv) > 2:
        result = hooks.on_file_create(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif command == "commit" and len(sys.argv) > 2:
        result = hooks.on_commit_attempt(sys.argv[2:])
        print(json.dumps(result, indent=2))

    elif command == "context" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        cursor_pos = None
        if len(sys.argv) > 3:
            try:
                cursor_pos = {"line": int(sys.argv[3]), "character": 0}
            except ValueError:
                pass

        # Simulate tab opening to get context
        tab_id = f"cli_tab_{hash(file_path)}"
        result = on_tab_opened(tab_id, file_path, cursor_pos, "cli_user")

        if result["status"] == "success":
            print("Context for file:")
            print(f"File: {file_path}")
            print(f"Relevance Score: {result.get('relevance_score', 0):.2f}")
            print("\nRelevant Patterns:")
            for pattern in result["context"].get("relevant_patterns", []):
                print(f"  - {pattern.get('pattern_type', 'unknown')}: {pattern.get('signature', '')[:50]}...")
            print("\nUnderstanding Context:")
            for key, value in result["context"].get("understanding_context", {}).items():
                print(f"  - {key}: {str(value)[:100]}...")
        else:
            print(f"Error getting context: {result}")

    else:
        print("Invalid command")
