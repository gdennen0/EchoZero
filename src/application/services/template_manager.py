"""
Template Manager

Handles the production template lifecycle:
- Locating the bundled template inside the application package
- Copying the template to a user-local working project (copy-on-first-use)
- Validating template contents
- Checking template version compatibility
- Exporting the current project as a new production template
"""
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.utils.message import Log
from src.utils.paths import get_template_path, get_production_project_path, get_app_install_dir


class TemplateValidationResult:
    """Result of template validation with issues list."""

    def __init__(self, valid: bool, issues: Optional[List[str]] = None):
        self.valid = valid
        self.issues = issues or []

    def __bool__(self) -> bool:
        return self.valid


class TemplateManager:
    """
    Manages the production template and the user's working copy.

    The template is a normal ``.ez`` project file bundled inside the
    application package at ``data/production_template.ez``.  On first
    production-mode launch the template is copied to a user-local
    path and loaded via the standard project loading infrastructure.
    """

    TEMPLATE_VERSION_KEY = "template_version"

    def __init__(self, facade=None):
        self._facade = facade

    # ------------------------------------------------------------------
    # Template location
    # ------------------------------------------------------------------

    @staticmethod
    def get_bundled_template() -> Optional[Path]:
        """Return path to bundled template, or None if absent."""
        return get_template_path()

    @staticmethod
    def get_working_copy() -> Path:
        """Return path to the user-local working-copy project."""
        return get_production_project_path()

    def has_bundled_template(self) -> bool:
        return self.get_bundled_template() is not None

    def has_working_copy(self) -> bool:
        return self.get_working_copy().is_file()

    # ------------------------------------------------------------------
    # Copy-on-first-use
    # ------------------------------------------------------------------

    def ensure_working_copy(self) -> Optional[Path]:
        """
        Ensure a working copy exists.  Copies from the bundled template
        if the working copy does not yet exist.

        Returns:
            Path to the working copy, or None on failure.
        """
        working = self.get_working_copy()
        if working.is_file():
            return working

        template = self.get_bundled_template()
        if template is None:
            Log.warning("TemplateManager: No bundled template found")
            return None

        try:
            working.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(template), str(working))
            Log.info(f"TemplateManager: Copied template to {working}")
            return working
        except Exception as exc:
            Log.error(f"TemplateManager: Failed to copy template: {exc}")
            return None

    # ------------------------------------------------------------------
    # Version checking
    # ------------------------------------------------------------------

    def get_template_version(self, ez_path: Path) -> Optional[str]:
        """Extract template_version from an .ez file's project.json."""
        try:
            if not zipfile.is_zipfile(str(ez_path)):
                return None
            with zipfile.ZipFile(str(ez_path), 'r') as zf:
                if 'project.json' not in zf.namelist():
                    return None
                data = json.loads(zf.read('project.json'))
                metadata = data.get('metadata', {})
                return metadata.get(self.TEMPLATE_VERSION_KEY)
        except Exception as exc:
            Log.warning(f"TemplateManager: Error reading template version: {exc}")
            return None

    def is_update_available(self) -> bool:
        """Check if the bundled template is newer than the working copy."""
        template = self.get_bundled_template()
        working = self.get_working_copy()
        if template is None or not working.is_file():
            return False
        bundled_ver = self.get_template_version(template)
        working_ver = self.get_template_version(working)
        if bundled_ver is None:
            return False
        return bundled_ver != working_ver

    def reset_to_template(self) -> Optional[Path]:
        """Delete the working copy and re-copy from the bundled template."""
        working = self.get_working_copy()
        if working.is_file():
            try:
                working.unlink()
            except Exception as exc:
                Log.error(f"TemplateManager: Failed to delete working copy: {exc}")
                return None
        return self.ensure_working_copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_template(self, ez_path: Path) -> TemplateValidationResult:
        """
        Validate that an .ez file is suitable as a production template.

        Checks:
        - File is a valid ZIP / .ez archive
        - Contains project.json
        - Has at least one SetlistAudioInput block
        - Has at least one Editor block
        - Has at least one action set with actions
        - All connections reference valid block IDs
        """
        issues: List[str] = []

        if not ez_path.is_file():
            return TemplateValidationResult(False, ["File does not exist"])

        if not zipfile.is_zipfile(str(ez_path)):
            return TemplateValidationResult(False, ["Not a valid .ez archive"])

        try:
            with zipfile.ZipFile(str(ez_path), 'r') as zf:
                if 'project.json' not in zf.namelist():
                    return TemplateValidationResult(False, ["Missing project.json"])
                data = json.loads(zf.read('project.json'))
        except Exception as exc:
            return TemplateValidationResult(False, [f"Error reading archive: {exc}"])

        blocks = data.get('blocks', [])
        connections = data.get('connections', [])
        action_sets = data.get('action_sets', [])
        action_items = data.get('action_items', [])

        block_ids = {b.get('id') for b in blocks if b.get('id')}
        block_types = {b.get('id'): b.get('type') for b in blocks}

        # Required block types
        has_audio_input = any(b.get('type') == 'SetlistAudioInput' for b in blocks)
        has_editor = any(b.get('type') == 'Editor' for b in blocks)

        if not has_audio_input:
            issues.append("Missing required block: SetlistAudioInput")
        if not has_editor:
            issues.append("Missing required block: Editor")

        # Action set validation
        if not action_sets and not action_items:
            issues.append("No action sets defined")
        elif action_items:
            has_set_audio = any(
                ai.get('action_name') == 'Set Audio File'
                for ai in action_items
            )
            if not has_set_audio:
                issues.append("Action set missing 'Set Audio File' action")

        # Connection validation
        for conn in connections:
            src = conn.get('source_block_id')
            tgt = conn.get('target_block_id')
            if src and src not in block_ids:
                issues.append(f"Connection references missing source block: {src}")
            if tgt and tgt not in block_ids:
                issues.append(f"Connection references missing target block: {tgt}")

        return TemplateValidationResult(len(issues) == 0, issues)

    # ------------------------------------------------------------------
    # Export current project as template
    # ------------------------------------------------------------------

    def export_as_template(self, target_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Save the current project as the production template.

        Args:
            target_path: Where to save.  Defaults to ``data/production_template.ez``
                         in the source tree.

        Returns:
            (success, message) tuple.
        """
        if self._facade is None:
            return False, "No facade available"

        if target_path is None:
            install_dir = get_app_install_dir()
            if install_dir is None:
                return False, "Cannot determine install directory"
            target_path = install_dir / "data" / "production_template.ez"

        # Save current project to a temp location first, then copy
        try:
            result = self._facade.save_project()
            if not result.success:
                return False, f"Failed to save project: {result.message}"

            # Find the saved .ez file
            project_id = self._facade.get_current_project_id()
            if not project_id:
                return False, "No current project"

            project = self._facade.project_service.load_project(project_id)
            if not project or not project.save_directory:
                return False, "Project has no save path"

            source_file = Path(project.save_directory) / f"{project.name}.ez"
            if not source_file.is_file():
                # Try alternative path patterns
                for f in Path(project.save_directory).glob("*.ez"):
                    source_file = f
                    break
                else:
                    return False, f"Cannot find saved .ez file in {project.save_directory}"

            # Validate before copying
            validation = self.validate_template(source_file)
            if not validation:
                issue_text = "\n".join(f"  - {i}" for i in validation.issues)
                return False, f"Template validation failed:\n{issue_text}"

            # Copy to target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_file), str(target_path))
            Log.info(f"TemplateManager: Exported template to {target_path}")
            return True, f"Template exported to {target_path}"

        except Exception as exc:
            return False, f"Export failed: {exc}"

    # ------------------------------------------------------------------
    # Load template into facade
    # ------------------------------------------------------------------

    def load_production_project(self) -> Tuple[bool, str]:
        """
        Ensure the working copy exists and load it via the facade.

        Returns:
            (success, message) tuple.
        """
        if self._facade is None:
            return False, "No facade available"

        working = self.ensure_working_copy()
        if working is None:
            return False, "No production template available"

        result = self._facade.load_project(str(working))
        if result.success:
            return True, "Production project loaded"
        return False, f"Failed to load production project: {result.message}"
