#!/usr/bin/env python3
"""
Cursor IDE Startup Script for EchoZero.

This script runs automatically when the EchoZero project is opened in Cursor IDE,
ensuring AgentAssets are synchronized and the development environment is ready.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path


def main():
    """Main startup routine for EchoZero in Cursor IDE."""
    project_root = Path(__file__).resolve().parent.parent
    agent_assets_root = project_root / "AgentAssets"

    print("🚀 EchoZero Cursor IDE Startup")
    print("=" * 50)

    # Check if we're in the right directory
    if not (project_root / "main_qt.py").exists():
        print("❌ Error: Not in EchoZero project root")
        return 1

    # Ensure AgentAssets directory exists
    if not agent_assets_root.exists():
        print("❌ Error: AgentAssets directory not found")
        return 1

    results = {}

    # 1. Validate AgentAssets integrity
    print("1. Validating AgentAssets integrity...")
    results["integrity"] = validate_agent_assets_integrity(agent_assets_root)

    # 2. Run initial synchronization
    print("2. Synchronizing AgentAssets...")
    results["sync"] = run_initial_sync(agent_assets_root, project_root)

    # 3. Check development environment
    print("3. Checking development environment...")
    results["environment"] = check_development_environment(project_root)

    # 4. Validate current codebase state
    print("4. Validating codebase state...")
    results["validation"] = validate_codebase_state(agent_assets_root, project_root)

    # 5. Initialize learning engine and context injection
    print("5. Starting learning engine and context injection...")
    results["learning"] = initialize_learning_system(agent_assets_root)

    # 6. Set up file watching (background)
    print("6. Starting file watcher...")
    results["watcher"] = start_file_watcher(agent_assets_root, project_root)

    # Report results
    print("\n" + "=" * 50)
    print("🎯 Startup Complete")
    print("=" * 50)

    all_success = True
    for check, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {check.title()}: {result['message']}")
        if not result["success"]:
            all_success = False

    if all_success:
        print("\n🎉 EchoZero is ready for AI agent development!")
        print("\n💡 Quick Actions:")
        print("   • Run 'AgentAssets: Sync' task to manual sync")
        print("   • Use 'AgentAssets: Quality Check' on files")
        print("   • Run 'AgentAssets: Watch Mode' for continuous sync")
        print("   • Check 'AgentAssets: Performance Report' for feedback")
    else:
        print("\n⚠️  Some checks failed. Run 'AgentAssets: Sync' task to fix.")

    return 0 if all_success else 1


def validate_agent_assets_integrity(agent_assets_root: Path) -> dict:
    """Validate AgentAssets directory structure and files."""
    try:
        required_files = [
            "core/CORE_VALUES.md",
            "core/CURRENT_STATE.md",
            "MODULE_INDEX.md",
            "scripts/quality_checks.py",
            "scripts/auto_sync.py",
            "scripts/ai_feedback_system.py"
        ]

        missing_files = []
        for file_path in required_files:
            full_path = agent_assets_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            return {
                "success": False,
                "message": f"Missing files: {', '.join(missing_files)}"
            }

        # Check script executability
        scripts_dir = agent_assets_root / "scripts"
        non_executable = []
        for script in scripts_dir.glob("*.py"):
            if not os.access(script, os.X_OK):
                try:
                    script.chmod(script.stat().st_mode | 0o111)
                except:
                    non_executable.append(script.name)

        if non_executable:
            return {
                "success": False,
                "message": f"Scripts not executable: {', '.join(non_executable)}"
            }

        return {
            "success": True,
            "message": "All AgentAssets files present and accessible"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Integrity check failed: {e}"
        }


def run_initial_sync(agent_assets_root: Path, project_root: Path) -> dict:
    """Run initial AgentAssets synchronization."""
    try:
        script_path = agent_assets_root / "scripts" / "auto_sync.py"

        result = subprocess.run([
            sys.executable, str(script_path), "sync"
        ], capture_output=True, text=True, cwd=project_root, timeout=60)

        if result.returncode == 0:
            return {
                "success": True,
                "message": "AgentAssets synchronized successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Sync failed: {result.stderr.strip()}"
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": "Sync timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Sync error: {e}"
        }


def check_development_environment(project_root: Path) -> dict:
    """Check that the development environment is properly set up."""
    try:
        issues = []

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 10):
            issues.append(f"Python {python_version.major}.{python_version.minor} < 3.10 required")

        # Check if requirements.txt exists and is readable
        requirements_path = project_root / "requirements.txt"
        if not requirements_path.exists():
            issues.append("requirements.txt not found")

        # Check if virtual environment is activated (basic check)
        in_venv = sys.prefix != sys.base_prefix
        if not in_venv:
            issues.append("Virtual environment not activated")

        # Check basic imports
        try:
            import pathlib
            import json
        except ImportError as e:
            issues.append(f"Basic imports failed: {e}")

        if issues:
            return {
                "success": False,
                "message": f"Environment issues: {'; '.join(issues)}"
            }

        return {
            "success": True,
            "message": "Development environment ready"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Environment check failed: {e}"
        }


def validate_codebase_state(agent_assets_root: Path, project_root: Path) -> dict:
    """Validate the current state of the codebase."""
    try:
        script_path = agent_assets_root / "scripts" / "quality_checks.py"

        # Run quality checks on a few key files
        key_files = [
            "src/application/blocks/registry.py",
            "src/domain/entities/block.py",
            "main_qt.py"
        ]

        issues_found = 0
        for file_path in key_files:
            full_path = project_root / file_path
            if full_path.exists():
                result = subprocess.run([
                    sys.executable, str(script_path), str(full_path)
                ], capture_output=True, text=True, cwd=project_root)

                if result.returncode != 0:
                    issues_found += 1

        if issues_found > 0:
            return {
                "success": False,
                "message": f"Quality issues found in {issues_found} key files"
            }

        return {
            "success": True,
            "message": "Codebase quality checks passed"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Validation error: {e}"
        }


def initialize_learning_system(agent_assets_root: Path) -> dict:
    """Initialize the learning engine and context injection system."""
    try:
        # Import and initialize context injector
        sys.path.insert(0, str(agent_assets_root / "scripts"))
        from context_injector import start_context_injection

        context_started = start_context_injection()

        if context_started:
            return {
                "success": True,
                "message": "Learning engine and context injection initialized"
            }
        else:
            return {
                "success": False,
                "message": "Failed to start context injection"
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Learning system initialization failed: {e}"
        }


def start_file_watcher(agent_assets_root: Path, project_root: Path) -> dict:
    """Start the file watcher in background mode."""
    try:
        # Check if watch mode is already running (simplified check)
        watch_pid_file = agent_assets_root / "data" / "watch.pid"
        if watch_pid_file.exists():
            try:
                with open(watch_pid_file, 'r') as f:
                    pid = int(f.read().strip())

                # Check if process is still running
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                return {
                    "success": True,
                    "message": "File watcher already running"
                }
            except (OSError, ValueError):
                # Process not running, remove stale PID file
                watch_pid_file.unlink(missing_ok=True)

        # Start watcher in background
        script_path = agent_assets_root / "scripts" / "auto_sync.py"

        process = subprocess.Popen([
            sys.executable, str(script_path), "watch"
        ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Save PID for future checks
        watch_pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(watch_pid_file, 'w') as f:
            f.write(str(process.pid))

        return {
            "success": True,
            "message": "File watcher started in background"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to start file watcher: {e}"
        }


if __name__ == "__main__":
    sys.exit(main())
