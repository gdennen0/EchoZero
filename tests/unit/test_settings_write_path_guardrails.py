"""
Guardrails for settings write-path consistency.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_block_panels_do_not_call_facade_update_block_metadata_directly():
    """
    Block panels should use settings managers or CommandBus helpers.
    """
    panels_dir = PROJECT_ROOT / "ui" / "qt_gui" / "block_panels"
    offenders = []
    for path in panels_dir.rglob("*.py"):
        text = _read(path)
        if "facade.update_block_metadata(" in text or "self.facade.update_block_metadata(" in text:
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert not offenders, f"Direct update_block_metadata usage in block panels: {offenders}"


def test_node_editor_widgets_do_not_call_facade_update_block_metadata_directly():
    """
    Node editor settings widgets should also use command-based writes.
    """
    node_editor_dir = PROJECT_ROOT / "ui" / "qt_gui" / "node_editor"
    offenders = []
    for path in node_editor_dir.rglob("*.py"):
        text = _read(path)
        if "facade.update_block_metadata(" in text or "self.facade.update_block_metadata(" in text:
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert not offenders, f"Direct update_block_metadata usage in node editor: {offenders}"


def test_quick_actions_do_not_call_facade_update_block_metadata_directly():
    """
    Quick actions should use settings managers or metadata commands.
    """
    quick_actions = PROJECT_ROOT / "src" / "application" / "blocks" / "quick_actions.py"
    text = _read(quick_actions)
    assert "facade.update_block_metadata(" not in text


def test_block_panel_base_uses_centralized_block_updated_handler():
    """
    Event bus subscription should target the centralized base handler so
    per-panel custom overrides cannot reintroduce full reload loops.
    """
    base_file = PROJECT_ROOT / "ui" / "qt_gui" / "block_panels" / "block_panel_base.py"
    text = _read(base_file)
    assert 'subscribe("BlockUpdated", self._on_block_updated_base)' in text
