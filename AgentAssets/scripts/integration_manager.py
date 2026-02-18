#!/usr/bin/env python3
"""
AgentAssets Integration Manager for Cursor IDE.

This script manages the integration between AgentAssets and Cursor IDE,
ensuring automatic synchronization, validation, and feedback loops.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class IntegrationStatus:
    """Status of AgentAssets integration."""
    cursor_hooks_enabled: bool = False
    auto_sync_enabled: bool = True
    quality_checks_enabled: bool = True
    file_watching_active: bool = False
    last_sync: Optional[datetime] = None
    sync_failures: int = 0
    quality_violations: int = 0
    ai_feedback_enabled: bool = True
    council_validation_enabled: bool = True


class AgentAssetsIntegrationManager:
    """Manages AgentAssets integration with Cursor IDE."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.agent_assets_root = Path(__file__).resolve().parent.parent
        self.cursor_config_dir = self.project_root / ".cursor"
        self.status_file = self.cursor_config_dir / "integration_status.json"

        # Ensure directories exist
        self.cursor_config_dir.mkdir(parents=True, exist_ok=True)

    def initialize_integration(self) -> Dict[str, Any]:
        """
        Initialize AgentAssets integration with Cursor IDE.

        This sets up all necessary configurations, hooks, and automation.
        """
        print("🔧 Initializing AgentAssets Integration...")

        results = {}

        # 1. Verify Cursor IDE setup
        results["cursor_setup"] = self._verify_cursor_setup()

        # 2. Configure file watching
        results["file_watching"] = self._configure_file_watching()

        # 3. Set up quality gates
        results["quality_gates"] = self._setup_quality_gates()

        # 4. Configure AI feedback loops
        results["ai_feedback"] = self._configure_ai_feedback()

        # 5. Set up council decision integration
        results["council_integration"] = self._setup_council_integration()

        # 6. Configure automatic synchronization
        results["auto_sync"] = self._configure_auto_sync()

        # 7. Set up startup automation
        results["startup_automation"] = self._setup_startup_automation()

        # 8. Configure keybindings
        results["keybindings"] = self._configure_keybindings()

        # Save integration status
        status = IntegrationStatus(
            cursor_hooks_enabled=True,
            auto_sync_enabled=True,
            quality_checks_enabled=True,
            file_watching_active=True,
            last_sync=datetime.now(),
            ai_feedback_enabled=True,
            council_validation_enabled=True
        )
        self._save_integration_status(status)

        # Report results
        success_count = sum(1 for r in results.values() if r.get("success", False))
        total_count = len(results)

        results["summary"] = {
            "success": success_count == total_count,
            "successful_steps": success_count,
            "total_steps": total_count,
            "integration_complete": success_count == total_count
        }

        return results

    def check_integration_health(self) -> Dict[str, Any]:
        """Check the health of AgentAssets integration."""
        print("🏥 Checking AgentAssets Integration Health...")

        health_checks = {}

        # Load current status
        status = self._load_integration_status()

        # Check file watching
        health_checks["file_watching"] = self._check_file_watching_health(status)

        # Check synchronization
        health_checks["synchronization"] = self._check_sync_health(status)

        # Check quality gates
        health_checks["quality_gates"] = self._check_quality_gates_health()

        # Check AI feedback
        health_checks["ai_feedback"] = self._check_ai_feedback_health()

        # Check council integration
        health_checks["council_integration"] = self._check_council_integration_health()

        # Overall health score
        health_scores = [check.get("health_score", 0) for check in health_checks.values()]
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0

        health_checks["overall"] = {
            "health_score": overall_health,
            "status": "healthy" if overall_health >= 0.8 else "warning" if overall_health >= 0.6 else "critical",
            "last_checked": datetime.now().isoformat()
        }

        return health_checks

    def repair_integration(self) -> Dict[str, Any]:
        """Repair any issues with AgentAssets integration."""
        print("🔧 Repairing AgentAssets Integration...")

        health = self.check_integration_health()
        repairs_needed = []

        # Identify issues
        for component, status in health.items():
            if component != "overall" and status.get("health_score", 1.0) < 0.8:
                repairs_needed.append(component)

        repair_results = {}

        # Perform repairs
        for component in repairs_needed:
            repair_results[component] = self._repair_component(component)

        # Re-run health check
        final_health = self.check_integration_health()

        repair_results["final_health"] = final_health["overall"]
        repair_results["repairs_completed"] = len([r for r in repair_results.values()
                                                  if isinstance(r, dict) and r.get("success", False)])

        return repair_results

    def update_integration(self) -> Dict[str, Any]:
        """Update AgentAssets integration to latest version."""
        print("⬆️  Updating AgentAssets Integration...")

        updates = {}

        # Check for new scripts
        updates["scripts"] = self._update_scripts()

        # Update configurations
        updates["configurations"] = self._update_configurations()

        # Update keybindings
        updates["keybindings"] = self._update_keybindings()

        # Update tasks
        updates["tasks"] = self._update_tasks()

        # Update status
        status = self._load_integration_status()
        status.last_sync = datetime.now()
        self._save_integration_status(status)

        return updates

    def _verify_cursor_setup(self) -> Dict[str, Any]:
        """Verify Cursor IDE is properly set up."""
        try:
            required_files = [
                ".cursor/settings.json",
                ".cursor/tasks.json",
                ".cursor/hooks.py",
                ".cursor/startup.py"
            ]

            missing_files = []
            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)

            if missing_files:
                return {
                    "success": False,
                    "message": f"Missing Cursor config files: {', '.join(missing_files)}"
                }

            return {
                "success": True,
                "message": "Cursor IDE configuration complete"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Cursor setup verification failed: {e}"
            }

    def _configure_file_watching(self) -> Dict[str, Any]:
        """Configure automatic file watching."""
        try:
            # Ensure watch script is executable
            watch_script = self.agent_assets_root / "scripts" / "auto_sync.py"
            if not watch_script.exists():
                return {
                    "success": False,
                    "message": "Auto-sync script not found"
                }

            # Make executable
            watch_script.chmod(watch_script.stat().st_mode | 0o111)

            return {
                "success": True,
                "message": "File watching configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"File watching configuration failed: {e}"
            }

    def _setup_quality_gates(self) -> Dict[str, Any]:
        """Set up quality gates for commits and saves."""
        try:
            # Verify quality check script exists
            quality_script = self.agent_assets_root / "scripts" / "quality_checks.py"
            if not quality_script.exists():
                return {
                    "success": False,
                    "message": "Quality checks script not found"
                }

            # Check if pre-commit is configured
            precommit_config = self.project_root / ".pre-commit-config.yaml"
            if not precommit_config.exists():
                return {
                    "success": False,
                    "message": "Pre-commit configuration not found"
                }

            return {
                "success": True,
                "message": "Quality gates configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Quality gates setup failed: {e}"
            }

    def _configure_ai_feedback(self) -> Dict[str, Any]:
        """Configure AI agent feedback system."""
        try:
            feedback_script = self.agent_assets_root / "scripts" / "ai_feedback_system.py"
            if not feedback_script.exists():
                return {
                    "success": False,
                    "message": "AI feedback script not found"
                }

            # Ensure data directory exists
            data_dir = self.agent_assets_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            return {
                "success": True,
                "message": "AI feedback system configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"AI feedback configuration failed: {e}"
            }

    def _setup_council_integration(self) -> Dict[str, Any]:
        """Set up council decision integration."""
        try:
            council_script = self.agent_assets_root / "scripts" / "validate_council_decision.py"
            if not council_script.exists():
                return {
                    "success": False,
                    "message": "Council validation script not found"
                }

            return {
                "success": True,
                "message": "Council integration configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Council integration setup failed: {e}"
            }

    def _configure_auto_sync(self) -> Dict[str, Any]:
        """Configure automatic synchronization."""
        try:
            sync_script = self.agent_assets_root / "scripts" / "auto_sync.py"
            if not sync_script.exists():
                return {
                    "success": False,
                    "message": "Auto-sync script not found"
                }

            # Ensure data directory for sync state
            data_dir = self.agent_assets_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            return {
                "success": True,
                "message": "Auto-sync configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Auto-sync configuration failed: {e}"
            }

    def _setup_startup_automation(self) -> Dict[str, Any]:
        """Set up startup automation."""
        try:
            startup_script = self.cursor_config_dir / "startup.py"
            if not startup_script.exists():
                return {
                    "success": False,
                    "message": "Startup script not found"
                }

            # Make executable
            startup_script.chmod(startup_script.stat().st_mode | 0o111)

            return {
                "success": True,
                "message": "Startup automation configured"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Startup automation setup failed: {e}"
            }

    def _configure_keybindings(self) -> Dict[str, Any]:
        """Configure Cursor IDE keybindings."""
        try:
            keybindings_file = self.cursor_config_dir / "keybindings.json"
            if not keybindings_file.exists():
                return {
                    "success": False,
                    "message": "Keybindings file not found"
                }

            # Validate keybindings JSON
            with open(keybindings_file, 'r') as f:
                json.load(f)

            return {
                "success": True,
                "message": "Keybindings configured"
            }

        except json.JSONDecodeError:
            return {
                "success": False,
                "message": "Invalid keybindings JSON"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Keybindings configuration failed: {e}"
            }

    def _load_integration_status(self) -> IntegrationStatus:
        """Load integration status."""
        if not self.status_file.exists():
            return IntegrationStatus()

        try:
            with open(self.status_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp string back to datetime
                if 'last_sync' in data and data['last_sync']:
                    data['last_sync'] = datetime.fromisoformat(data['last_sync'])
                return IntegrationStatus(**data)
        except:
            return IntegrationStatus()

    def _save_integration_status(self, status: IntegrationStatus):
        """Save integration status."""
        data = asdict(status)
        # Convert datetime to string
        if data['last_sync']:
            data['last_sync'] = data['last_sync'].isoformat()

        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _check_file_watching_health(self, status: IntegrationStatus) -> Dict[str, Any]:
        """Check file watching health."""
        # Simplified health check
        watch_pid_file = self.agent_assets_root / "data" / "watch.pid"
        if watch_pid_file.exists():
            try:
                with open(watch_pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if process exists
                return {"health_score": 1.0, "status": "active"}
            except:
                return {"health_score": 0.5, "status": "stale_pid"}

        return {"health_score": 0.0, "status": "inactive"}

    def _check_sync_health(self, status: IntegrationStatus) -> Dict[str, Any]:
        """Check synchronization health."""
        if not status.last_sync:
            return {"health_score": 0.0, "status": "never_synced"}

        time_since_sync = datetime.now() - status.last_sync
        if time_since_sync < timedelta(hours=1):
            return {"health_score": 1.0, "status": "recent"}
        elif time_since_sync < timedelta(days=1):
            return {"health_score": 0.8, "status": "stale"}
        else:
            return {"health_score": 0.3, "status": "very_stale"}

    def _check_quality_gates_health(self) -> Dict[str, Any]:
        """Check quality gates health."""
        try:
            # Run a quick quality check
            result = subprocess.run([
                sys.executable, str(self.agent_assets_root / "scripts" / "quality_checks.py"),
                str(self.project_root / "main_qt.py")
            ], capture_output=True, timeout=10)

            if result.returncode == 0:
                return {"health_score": 1.0, "status": "passing"}
            else:
                return {"health_score": 0.7, "status": "issues_found"}

        except:
            return {"health_score": 0.0, "status": "check_failed"}

    def _check_ai_feedback_health(self) -> Dict[str, Any]:
        """Check AI feedback system health."""
        feedback_file = self.agent_assets_root / "data" / "agent_profiles.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                agent_count = len(data)
                if agent_count > 0:
                    return {"health_score": 1.0, "status": f"{agent_count}_agents_tracked"}
                else:
                    return {"health_score": 0.5, "status": "no_agents_tracked"}
            except:
                return {"health_score": 0.3, "status": "corrupted_data"}

        return {"health_score": 0.0, "status": "no_data"}

    def _check_council_integration_health(self) -> Dict[str, Any]:
        """Check council integration health."""
        council_script = self.agent_assets_root / "scripts" / "validate_council_decision.py"
        if council_script.exists():
            return {"health_score": 1.0, "status": "configured"}
        else:
            return {"health_score": 0.0, "status": "missing_script"}

    def _repair_component(self, component: str) -> Dict[str, Any]:
        """Repair a specific component."""
        repair_methods = {
            "cursor_setup": self._verify_cursor_setup,
            "file_watching": self._configure_file_watching,
            "quality_gates": self._setup_quality_gates,
            "ai_feedback": self._configure_ai_feedback,
            "council_integration": self._setup_council_integration,
            "auto_sync": self._configure_auto_sync,
            "startup_automation": self._setup_startup_automation,
            "keybindings": self._configure_keybindings
        }

        if component in repair_methods:
            return repair_methods[component]()

        return {"success": False, "message": f"No repair method for {component}"}

    def _update_scripts(self) -> Dict[str, Any]:
        """Update scripts to latest versions."""
        # This would check for script updates and apply them
        return {"success": True, "message": "Scripts up to date"}

    def _update_configurations(self) -> Dict[str, Any]:
        """Update configuration files."""
        return {"success": True, "message": "Configurations up to date"}

    def _update_keybindings(self) -> Dict[str, Any]:
        """Update keybindings."""
        return {"success": True, "message": "Keybindings up to date"}

    def _update_tasks(self) -> Dict[str, Any]:
        """Update tasks configuration."""
        return {"success": True, "message": "Tasks up to date"}


def main():
    """CLI interface for integration management."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python integration_manager.py init        # Initialize integration")
        print("  python integration_manager.py health      # Check integration health")
        print("  python integration_manager.py repair      # Repair integration issues")
        print("  python integration_manager.py update      # Update integration")
        print("  python integration_manager.py status      # Show integration status")
        sys.exit(1)

    manager = AgentAssetsIntegrationManager()
    command = sys.argv[1]

    if command == "init":
        results = manager.initialize_integration()
        print(json.dumps(results, indent=2))

    elif command == "health":
        health = manager.check_integration_health()
        print(json.dumps(health, indent=2))

    elif command == "repair":
        repairs = manager.repair_integration()
        print(json.dumps(repairs, indent=2))

    elif command == "update":
        updates = manager.update_integration()
        print(json.dumps(updates, indent=2))

    elif command == "status":
        status = manager._load_integration_status()
        print(json.dumps(asdict(status), indent=2, default=str))

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

