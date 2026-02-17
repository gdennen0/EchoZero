#!/usr/bin/env python3
"""
EchoZero Refactoring Tracker

A tool to help AI agents track, verify, and execute the refactoring plan properly,
completely, and thoroughly.

Usage:
    python refactoring_tracker.py status          # Show overall status
    python refactoring_tracker.py phase <n>       # Show phase n details
    python refactoring_tracker.py verify <task>   # Verify a specific task
    python refactoring_tracker.py complete <task> # Mark task as complete
    python refactoring_tracker.py next            # Show next recommended task
    python refactoring_tracker.py check           # Run all verification checks
"""

import json
import os
import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    VERIFIED = "verified"


@dataclass
class Task:
    """A single refactoring task."""
    id: str
    name: str
    description: str
    phase: int
    priority: str  # HIGH, MEDIUM, LOW
    effort: str  # e.g., "2-3 days"
    dependencies: List[str] = field(default_factory=list)
    files_to_create: List[str] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    files_to_delete: List[str] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.NOT_STARTED
    completed_date: Optional[str] = None
    verified_date: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "phase": self.phase,
            "priority": self.priority,
            "effort": self.effort,
            "dependencies": self.dependencies,
            "files_to_create": self.files_to_create,
            "files_to_modify": self.files_to_modify,
            "files_to_delete": self.files_to_delete,
            "verification_commands": self.verification_commands,
            "status": self.status.value,
            "completed_date": self.completed_date,
            "verified_date": self.verified_date,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            phase=data["phase"],
            priority=data["priority"],
            effort=data["effort"],
            dependencies=data.get("dependencies", []),
            files_to_create=data.get("files_to_create", []),
            files_to_modify=data.get("files_to_modify", []),
            files_to_delete=data.get("files_to_delete", []),
            verification_commands=data.get("verification_commands", []),
            status=TaskStatus(data.get("status", "not_started")),
            completed_date=data.get("completed_date"),
            verified_date=data.get("verified_date"),
            notes=data.get("notes", ""),
        )


class RefactoringTracker:
    """Track and manage the refactoring process."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Find project root (look for main_qt.py)
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "main_qt.py").exists():
                    project_root = parent
                    break
            if project_root is None:
                project_root = Path.cwd()
        
        self.project_root = project_root
        self.data_file = self.project_root / "AgentAssets" / "data" / "refactoring_progress.json"
        self.tasks: Dict[str, Task] = {}
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing progress or initialize with default tasks."""
        if self.data_file.exists():
            self._load()
        else:
            self._initialize_tasks()
            self._save()

    def _load(self):
        """Load progress from file."""
        with open(self.data_file, "r") as f:
            data = json.load(f)
            self.tasks = {
                task_id: Task.from_dict(task_data)
                for task_id, task_data in data.get("tasks", {}).items()
            }

    def _save(self):
        """Save progress to file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, "w") as f:
            json.dump(
                {"tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}},
                f,
                indent=2,
            )

    def _initialize_tasks(self):
        """Initialize all tasks from the master plan."""
        # Phase 1: Core Abstractions
        self.tasks["p1_settings_validation"] = Task(
            id="p1_settings_validation",
            name="Settings Validation",
            description="Add validate() method to BaseSettings",
            phase=1,
            priority="HIGH",
            effort="1 day",
            files_to_modify=["src/application/settings/base_settings.py"],
            verification_commands=[
                "python3 -c \"from src.application.settings.base_settings import BaseSettings; print('validate' in dir(BaseSettings))\""
            ],
        )

        self.tasks["p1_settings_registry"] = Task(
            id="p1_settings_registry",
            name="Settings Registry",
            description="Create SettingsRegistry for auto-discovery",
            phase=1,
            priority="HIGH",
            effort="1 day",
            dependencies=["p1_settings_validation"],
            files_to_create=["src/shared/application/settings/settings_registry.py"],
            verification_commands=[
                "python -c \"from src.shared.application.settings.settings_registry import SettingsRegistry; print('OK')\""
            ],
        )

        self.tasks["p1_base_repository"] = Task(
            id="p1_base_repository",
            name="BaseRepository Pattern",
            description="Create generic BaseRepository with common CRUD operations",
            phase=1,
            priority="HIGH",
            effort="2-3 days",
            files_to_create=["src/shared/infrastructure/persistence/base_repository.py"],
            verification_commands=[
                "python -c \"from src.shared.infrastructure.persistence.base_repository import BaseRepository; print('OK')\""
            ],
        )

        self.tasks["p1_status_publisher"] = Task(
            id="p1_status_publisher",
            name="StatusPublisher Pattern",
            description="Create unified status update pattern",
            phase=1,
            priority="HIGH",
            effort="3-4 days",
            files_to_create=["src/shared/application/status/status_publisher.py"],
            files_to_modify=["src/application/services/block_status_service.py"],
            verification_commands=[
                "python -c \"from src.shared.application.status.status_publisher import StatusPublisher; print('OK')\""
            ],
        )

        self.tasks["p1_module_registry"] = Task(
            id="p1_module_registry",
            name="ModuleRegistry Pattern",
            description="Create generic registry for all module types",
            phase=1,
            priority="MEDIUM",
            effort="2 days",
            files_to_create=["src/shared/infrastructure/registry/module_registry.py"],
            verification_commands=[
                "python -c \"from src.shared.infrastructure.registry.module_registry import ModuleRegistry; print('OK')\""
            ],
        )

        self.tasks["p1_validation_framework"] = Task(
            id="p1_validation_framework",
            name="Validation Framework",
            description="Create standard validation framework",
            phase=1,
            priority="MEDIUM",
            effort="2 days",
            files_to_create=["src/shared/application/validation/validation_framework.py"],
            verification_commands=[
                "python -c \"from src.shared.application.validation.validation_framework import ValidationResult; print('OK')\""
            ],
        )

        # Phase 2: Structure - Vertical Feature Modules
        self.tasks["p2_create_structure"] = Task(
            id="p2_create_structure",
            name="Create Base Structure",
            description="Create features/ and shared/ directories",
            phase=2,
            priority="HIGH",
            effort="1 day",
            dependencies=["p1_base_repository", "p1_status_publisher"],
            files_to_create=[
                "src/features/__init__.py",
                "src/shared/__init__.py",
                "ui/features/__init__.py",
                "ui/shared/__init__.py",
            ],
        )

        self.tasks["p2_migrate_shared"] = Task(
            id="p2_migrate_shared",
            name="Migrate Shared Code",
            description="Move cross-cutting concerns to shared/",
            phase=2,
            priority="HIGH",
            effort="2-3 days",
            dependencies=["p2_create_structure"],
        )

        self.tasks["p2_migrate_connections"] = Task(
            id="p2_migrate_connections",
            name="Migrate Connections Feature",
            description="Move connections feature (smallest test case)",
            phase=2,
            priority="HIGH",
            effort="1 day",
            dependencies=["p2_migrate_shared"],
        )

        self.tasks["p2_migrate_projects"] = Task(
            id="p2_migrate_projects",
            name="Migrate Projects Feature",
            description="Move projects feature",
            phase=2,
            priority="HIGH",
            effort="2 days",
            dependencies=["p2_migrate_connections"],
        )

        self.tasks["p2_migrate_blocks"] = Task(
            id="p2_migrate_blocks",
            name="Migrate Blocks Feature",
            description="Move blocks feature (largest)",
            phase=2,
            priority="HIGH",
            effort="4-5 days",
            dependencies=["p2_migrate_projects"],
        )

        self.tasks["p2_migrate_execution"] = Task(
            id="p2_migrate_execution",
            name="Migrate Execution Feature",
            description="Move execution feature",
            phase=2,
            priority="HIGH",
            effort="2 days",
            dependencies=["p2_migrate_blocks"],
        )

        self.tasks["p2_migrate_setlists"] = Task(
            id="p2_migrate_setlists",
            name="Migrate Setlists Feature",
            description="Move setlists feature",
            phase=2,
            priority="HIGH",
            effort="2 days",
            dependencies=["p2_migrate_execution"],
        )

        self.tasks["p2_migrate_ma3"] = Task(
            id="p2_migrate_ma3",
            name="Migrate MA3 Feature",
            description="Move MA3 feature",
            phase=2,
            priority="HIGH",
            effort="2 days",
            dependencies=["p2_migrate_setlists"],
        )

        self.tasks["p2_unified_api"] = Task(
            id="p2_unified_api",
            name="Create Unified API Facade",
            description="Split ApplicationFacade into feature facades",
            phase=2,
            priority="HIGH",
            effort="3-4 days",
            dependencies=["p2_migrate_ma3"],
        )

        self.tasks["p2_cleanup"] = Task(
            id="p2_cleanup",
            name="Phase 2 Cleanup",
            description="Remove old structure, verify imports, run tests",
            phase=2,
            priority="HIGH",
            effort="1-2 days",
            dependencies=["p2_unified_api"],
            verification_commands=[
                "python -m pytest tests/ -v",
                "python main_qt.py --help",
            ],
        )

        # Phase 3: Feature Refactors
        self.tasks["p3_editor_api"] = Task(
            id="p3_editor_api",
            name="Editor Unified API",
            description="Implement unified layer/event API",
            phase=3,
            priority="MEDIUM",
            effort="1-2 weeks",
            dependencies=["p2_cleanup"],
        )

        self.tasks["p3_layer_sync"] = Task(
            id="p3_layer_sync",
            name="Layer Sync Refactor",
            description="Refactor to explicit synced layers",
            phase=3,
            priority="MEDIUM",
            effort="2-3 weeks",
            dependencies=["p2_cleanup"],
        )

        self.tasks["p3_block_filter"] = Task(
            id="p3_block_filter",
            name="Block Filter Cleanup",
            description="Remove migration code, simplify filter logic",
            phase=3,
            priority="LOW",
            effort="1 week",
            dependencies=["p2_cleanup"],
        )

        self.tasks["p3_project_actions"] = Task(
            id="p3_project_actions",
            name="Project Actions Refactor",
            description="Add project-level actions to action sets",
            phase=3,
            priority="LOW",
            effort="1 week",
            dependencies=["p2_cleanup"],
        )

        # Phase 4: Cleanup
        self.tasks["p4_directory_naming"] = Task(
            id="p4_directory_naming",
            name="Directory Naming Fixes",
            description="Rename Command/, CLI/, Utils/ to lowercase",
            phase=4,
            priority="MEDIUM",
            effort="1 day",
            dependencies=["p3_editor_api", "p3_layer_sync", "p3_block_filter"],
        )

        self.tasks["p4_legacy_removal"] = Task(
            id="p4_legacy_removal",
            name="Legacy Code Removal",
            description="Remove legacy duplicates and unused code",
            phase=4,
            priority="MEDIUM",
            effort="1 day",
            dependencies=["p4_directory_naming"],
            files_to_delete=["src/Utils/settings.py"],
        )

        self.tasks["p4_documentation"] = Task(
            id="p4_documentation",
            name="Documentation Updates",
            description="Update all documentation with new paths",
            phase=4,
            priority="LOW",
            effort="2 days",
            dependencies=["p4_legacy_removal"],
        )

    def get_status(self) -> Dict[str, Any]:
        """Get overall refactoring status."""
        phases = {1: [], 2: [], 3: [], 4: []}
        for task in self.tasks.values():
            phases[task.phase].append(task)

        status = {
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks.values() if t.status in [TaskStatus.COMPLETED, TaskStatus.VERIFIED]),
            "in_progress": sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "not_started": sum(1 for t in self.tasks.values() if t.status == TaskStatus.NOT_STARTED),
            "blocked": sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED),
            "phases": {},
        }

        for phase_num, tasks in phases.items():
            phase_status = {
                "total": len(tasks),
                "completed": sum(1 for t in tasks if t.status in [TaskStatus.COMPLETED, TaskStatus.VERIFIED]),
                "in_progress": sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
                "tasks": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "status": t.status.value,
                        "priority": t.priority,
                    }
                    for t in sorted(tasks, key=lambda x: (x.status != TaskStatus.IN_PROGRESS, x.priority != "HIGH", x.id))
                ],
            }
            status["phases"][phase_num] = phase_status

        return status

    def get_phase_details(self, phase: int) -> Dict[str, Any]:
        """Get detailed information about a phase."""
        tasks = [t for t in self.tasks.values() if t.phase == phase]
        return {
            "phase": phase,
            "tasks": [t.to_dict() for t in sorted(tasks, key=lambda x: x.id)],
        }

    def get_next_task(self) -> Optional[Task]:
        """Get the next recommended task to work on."""
        # First, check for in-progress tasks
        in_progress = [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        if in_progress:
            return in_progress[0]

        # Find tasks whose dependencies are all completed
        available = []
        for task in self.tasks.values():
            if task.status == TaskStatus.NOT_STARTED:
                deps_met = all(
                    self.tasks[dep].status in [TaskStatus.COMPLETED, TaskStatus.VERIFIED]
                    for dep in task.dependencies
                    if dep in self.tasks
                )
                if deps_met:
                    available.append(task)

        if not available:
            return None

        # Sort by phase, then priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        available.sort(key=lambda t: (t.phase, priority_order.get(t.priority, 3)))
        return available[0]

    def start_task(self, task_id: str) -> bool:
        """Mark a task as in progress."""
        if task_id not in self.tasks:
            return False
        self.tasks[task_id].status = TaskStatus.IN_PROGRESS
        self._save()
        return True

    def complete_task(self, task_id: str, notes: str = "") -> bool:
        """Mark a task as completed."""
        if task_id not in self.tasks:
            return False
        self.tasks[task_id].status = TaskStatus.COMPLETED
        self.tasks[task_id].completed_date = datetime.now().isoformat()
        if notes:
            self.tasks[task_id].notes = notes
        self._save()
        return True

    def verify_task(self, task_id: str) -> Dict[str, Any]:
        """Run verification for a task."""
        if task_id not in self.tasks:
            return {"success": False, "error": f"Task {task_id} not found"}

        task = self.tasks[task_id]
        results = {"task_id": task_id, "checks": [], "all_passed": True}

        # Check files to create exist
        for file_path in task.files_to_create:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            results["checks"].append({
                "type": "file_exists",
                "path": file_path,
                "passed": exists,
            })
            if not exists:
                results["all_passed"] = False

        # Check files to delete are gone
        for file_path in task.files_to_delete:
            full_path = self.project_root / file_path
            deleted = not full_path.exists()
            results["checks"].append({
                "type": "file_deleted",
                "path": file_path,
                "passed": deleted,
            })
            if not deleted:
                results["all_passed"] = False

        # Run verification commands
        for cmd in task.verification_commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=30,
                )
                passed = result.returncode == 0
                results["checks"].append({
                    "type": "command",
                    "command": cmd,
                    "passed": passed,
                    "stdout": result.stdout[:500] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else "",
                })
                if not passed:
                    results["all_passed"] = False
            except subprocess.TimeoutExpired:
                results["checks"].append({
                    "type": "command",
                    "command": cmd,
                    "passed": False,
                    "error": "Timeout",
                })
                results["all_passed"] = False
            except Exception as e:
                results["checks"].append({
                    "type": "command",
                    "command": cmd,
                    "passed": False,
                    "error": str(e),
                })
                results["all_passed"] = False

        # Update task status if all checks passed
        if results["all_passed"] and task.status == TaskStatus.COMPLETED:
            task.status = TaskStatus.VERIFIED
            task.verified_date = datetime.now().isoformat()
            self._save()

        return results

    def run_all_checks(self) -> Dict[str, Any]:
        """Run verification checks for all tasks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
            "summary": {"passed": 0, "failed": 0, "skipped": 0},
        }

        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.VERIFIED]:
                task_results = self.verify_task(task_id)
                results["tasks"][task_id] = task_results
                if task_results["all_passed"]:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
            else:
                results["summary"]["skipped"] += 1

        return results

    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get full context for working on a task."""
        if task_id not in self.tasks:
            return {"error": f"Task {task_id} not found"}

        task = self.tasks[task_id]
        
        # Get dependency status
        deps_status = []
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep = self.tasks[dep_id]
                deps_status.append({
                    "id": dep_id,
                    "name": dep.name,
                    "status": dep.status.value,
                })

        # Find what this task blocks
        blocks = []
        for other_id, other_task in self.tasks.items():
            if task_id in other_task.dependencies:
                blocks.append({
                    "id": other_id,
                    "name": other_task.name,
                    "status": other_task.status.value,
                })

        return {
            "task": task.to_dict(),
            "dependencies": deps_status,
            "blocks": blocks,
            "ready_to_start": all(
                self.tasks[dep].status in [TaskStatus.COMPLETED, TaskStatus.VERIFIED]
                for dep in task.dependencies
                if dep in self.tasks
            ),
        }


def print_status(tracker: RefactoringTracker):
    """Print overall status."""
    status = tracker.get_status()
    print("\n" + "=" * 60)
    print("ECHOZERO REFACTORING STATUS")
    print("=" * 60)
    print(f"\nTotal Tasks: {status['total_tasks']}")
    print(f"  Completed: {status['completed']}")
    print(f"  In Progress: {status['in_progress']}")
    print(f"  Not Started: {status['not_started']}")
    print(f"  Blocked: {status['blocked']}")
    
    for phase_num, phase_data in sorted(status["phases"].items()):
        print(f"\n{'─' * 60}")
        print(f"Phase {phase_num}: {phase_data['completed']}/{phase_data['total']} complete")
        print("─" * 60)
        for task in phase_data["tasks"]:
            status_icon = {
                "not_started": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "verified": "[v]",
                "blocked": "[!]",
            }.get(task["status"], "[ ]")
            print(f"  {status_icon} [{task['priority'][0]}] {task['name']} ({task['id']})")


def print_next_task(tracker: RefactoringTracker):
    """Print the next recommended task."""
    task = tracker.get_next_task()
    if task:
        print("\n" + "=" * 60)
        print("NEXT RECOMMENDED TASK")
        print("=" * 60)
        print(f"\nID: {task.id}")
        print(f"Name: {task.name}")
        print(f"Description: {task.description}")
        print(f"Phase: {task.phase}")
        print(f"Priority: {task.priority}")
        print(f"Estimated Effort: {task.effort}")
        
        if task.dependencies:
            print(f"\nDependencies:")
            for dep in task.dependencies:
                if dep in tracker.tasks:
                    dep_task = tracker.tasks[dep]
                    print(f"  - {dep}: {dep_task.status.value}")
        
        if task.files_to_create:
            print(f"\nFiles to Create:")
            for f in task.files_to_create:
                print(f"  - {f}")
        
        if task.files_to_modify:
            print(f"\nFiles to Modify:")
            for f in task.files_to_modify:
                print(f"  - {f}")
        
        print(f"\nTo start this task: python refactoring_tracker.py start {task.id}")
    else:
        print("\nNo available tasks. All tasks may be completed or blocked.")


def print_task_context(tracker: RefactoringTracker, task_id: str):
    """Print full context for a task."""
    context = tracker.get_task_context(task_id)
    if "error" in context:
        print(f"\nError: {context['error']}")
        return

    task = context["task"]
    print("\n" + "=" * 60)
    print(f"TASK CONTEXT: {task['name']}")
    print("=" * 60)
    
    print(f"\nID: {task['id']}")
    print(f"Status: {task['status']}")
    print(f"Phase: {task['phase']}")
    print(f"Priority: {task['priority']}")
    print(f"Effort: {task['effort']}")
    print(f"Description: {task['description']}")
    
    print(f"\nReady to Start: {'Yes' if context['ready_to_start'] else 'No'}")
    
    if context["dependencies"]:
        print(f"\nDependencies:")
        for dep in context["dependencies"]:
            status_icon = "[x]" if dep["status"] in ["completed", "verified"] else "[ ]"
            print(f"  {status_icon} {dep['name']} ({dep['status']})")
    
    if context["blocks"]:
        print(f"\nBlocks:")
        for blocked in context["blocks"]:
            print(f"  - {blocked['name']} ({blocked['status']})")
    
    if task["files_to_create"]:
        print(f"\nFiles to Create:")
        for f in task["files_to_create"]:
            exists = (tracker.project_root / f).exists()
            status_icon = "[x]" if exists else "[ ]"
            print(f"  {status_icon} {f}")
    
    if task["files_to_modify"]:
        print(f"\nFiles to Modify:")
        for f in task["files_to_modify"]:
            print(f"  - {f}")
    
    if task["files_to_delete"]:
        print(f"\nFiles to Delete:")
        for f in task["files_to_delete"]:
            deleted = not (tracker.project_root / f).exists()
            status_icon = "[x]" if deleted else "[ ]"
            print(f"  {status_icon} {f}")


def main():
    tracker = RefactoringTracker()

    if len(sys.argv) < 2:
        print("Usage: python refactoring_tracker.py <command> [args]")
        print("\nCommands:")
        print("  status          Show overall status")
        print("  phase <n>       Show phase n details")
        print("  next            Show next recommended task")
        print("  context <id>    Show full context for a task")
        print("  start <id>      Mark task as in progress")
        print("  complete <id>   Mark task as completed")
        print("  verify <id>     Verify a specific task")
        print("  check           Run all verification checks")
        return

    command = sys.argv[1].lower()

    if command == "status":
        print_status(tracker)

    elif command == "phase":
        if len(sys.argv) < 3:
            print("Usage: python refactoring_tracker.py phase <n>")
            return
        phase = int(sys.argv[2])
        details = tracker.get_phase_details(phase)
        print(f"\nPhase {phase} Tasks:")
        print("=" * 60)
        for task in details["tasks"]:
            print(f"\n{task['name']} ({task['id']})")
            print(f"  Status: {task['status']}")
            print(f"  Priority: {task['priority']}")
            print(f"  Effort: {task['effort']}")

    elif command == "next":
        print_next_task(tracker)

    elif command == "context":
        if len(sys.argv) < 3:
            print("Usage: python refactoring_tracker.py context <task_id>")
            return
        print_task_context(tracker, sys.argv[2])

    elif command == "start":
        if len(sys.argv) < 3:
            print("Usage: python refactoring_tracker.py start <task_id>")
            return
        task_id = sys.argv[2]
        if tracker.start_task(task_id):
            print(f"Task {task_id} marked as in progress")
        else:
            print(f"Task {task_id} not found")

    elif command == "complete":
        if len(sys.argv) < 3:
            print("Usage: python refactoring_tracker.py complete <task_id>")
            return
        task_id = sys.argv[2]
        notes = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        if tracker.complete_task(task_id, notes):
            print(f"Task {task_id} marked as completed")
            # Auto-verify
            results = tracker.verify_task(task_id)
            if results["all_passed"]:
                print(f"Task {task_id} verified successfully")
            else:
                print(f"Task {task_id} verification failed - check results")
        else:
            print(f"Task {task_id} not found")

    elif command == "verify":
        if len(sys.argv) < 3:
            print("Usage: python refactoring_tracker.py verify <task_id>")
            return
        task_id = sys.argv[2]
        results = tracker.verify_task(task_id)
        print(f"\nVerification Results for {task_id}:")
        print("=" * 60)
        for check in results["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['type']}: {check.get('path') or check.get('command', '')[:50]}")
            if not check["passed"] and check.get("error"):
                print(f"        Error: {check['error']}")
        print(f"\nOverall: {'PASSED' if results['all_passed'] else 'FAILED'}")

    elif command == "check":
        results = tracker.run_all_checks()
        print("\nVerification Results:")
        print("=" * 60)
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Skipped: {results['summary']['skipped']}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
