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
    allowed = {
        "ui/qt_gui/block_panels/block_panel_base.py",
    }
    offenders = []
    for path in panels_dir.rglob("*.py"):
        text = _read(path)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in allowed:
            continue
        if "facade.update_block_metadata(" in text or "self.facade.update_block_metadata(" in text:
            offenders.append(rel)
    assert not offenders, f"Direct update_block_metadata usage in block panels: {offenders}"


def test_node_editor_widgets_do_not_call_facade_update_block_metadata_directly():
    """
    Node editor settings widgets should also use command-based writes.
    """
    node_editor_dir = PROJECT_ROOT / "ui" / "qt_gui" / "node_editor"
    allowed = {
        "ui/qt_gui/node_editor/block_item.py",
    }
    offenders = []
    for path in node_editor_dir.rglob("*.py"):
        text = _read(path)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in allowed:
            continue
        if "facade.update_block_metadata(" in text or "self.facade.update_block_metadata(" in text:
            offenders.append(rel)
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


def test_show_manager_uses_commit_on_complete_for_key_inputs():
    """
    ShowManager key inputs should commit on completed edits, not intermediate changes.
    """
    file_path = PROJECT_ROOT / "ui" / "qt_gui" / "block_panels" / "show_manager_panel.py"
    text = _read(file_path)

    # Target TC text input must be wired.
    assert "self.target_timecode_edit.editingFinished.connect(self._on_target_timecode_changed)" in text

    # Port spinboxes in settings dialog should not use valueChanged for persistence.
    assert "self.ma3_port_spin.valueChanged.connect(self._on_port_changed)" not in text
    assert "self.listen_port_spin.valueChanged.connect(self._on_listen_port_changed)" not in text

    # Table spinboxes should use editingFinished commit handlers.
    assert "tg_spin.editingFinished.connect(" in text
    assert "seq_spin.editingFinished.connect(" in text
