"""Visual Lab tests: focused coverage for the external preview harness.
Exists to prove token loading, scene construction, and fallback capture behavior.
These tests avoid full EchoZero app workflow and keep screenshots under temp paths.
"""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QSpinBox

from dev.visual_lab.capture import capture_widget
from dev.visual_lab.current_state import (
    DRUMS_LAYER_ID,
    SELECTED_EVENT_ID,
    build_current_visual_lab_state,
)
from dev.visual_lab.font_assets import import_lab_fonts, list_lab_font_assets
from dev.visual_lab.preview import build_preview, build_visual_lab_catalog
from dev.visual_lab.scenes import build_visual_lab_presentation
from dev.visual_lab.token_editor import ColorTokenEditor, FontFamilyEditor
from dev.visual_lab.tokens import (
    TokenFieldSpec,
    VisualLabFonts,
    VisualLabGlobalColors,
    VisualLabMetrics,
    VisualLabPalette,
    load_tokens,
    save_tokens,
    token_field_specs,
    update_token_values,
)
from dev.visual_lab.waveforms import (
    FUN_WAVEFORM_KEY,
    build_fun_event_preview_state,
    build_fun_waveform_peaks,
    register_fun_waveform_preview,
)
from echozero.ui.qt.timeline.waveform_cache import get_cached_waveform


def test_visual_lab_tokens_load_editable_vocabulary():
    tokens = load_tokens()

    assert tokens.palette.window.startswith("#")
    assert tokens.global_colors.app_background.startswith("#")
    assert tokens.global_colors.primary == tokens.palette.accent
    assert tokens.metrics.audio_row_height_px > tokens.metrics.stem_row_height_px
    assert tokens.metrics.glow_strength_px >= 0
    assert tokens.fonts.base_px >= tokens.fonts.small_px


def test_visual_lab_token_editor_schema_exposes_all_knobs():
    tokens = load_tokens()
    specs = token_field_specs(tokens)
    expected_count = (
        len(fields(VisualLabGlobalColors))
        + len(fields(VisualLabPalette))
        + len(fields(VisualLabFonts))
        + len(fields(VisualLabMetrics))
    )

    assert len(specs) == expected_count
    assert {spec.section for spec in specs} == {
        "global_colors",
        "palette",
        "fonts",
        "metrics",
    }
    assert {spec.path for spec in specs} >= {
        "global_colors.primary",
        "palette.waveform",
        "palette.status_stale",
        "fonts.title_px",
        "metrics.corner_radius_px",
        "metrics.audio_row_height_px",
        "metrics.waveform_height_px",
    }


def test_visual_lab_token_updates_validate_and_roundtrip(tmp_path):
    tokens = load_tokens()
    updated = update_token_values(
        tokens,
        (
            ("global_colors.primary", "#223344"),
            ("palette.window", "#112233"),
            ("fonts.base_px", 15),
            ("metrics.corner_radius_px", "9"),
        ),
    )
    output_path = tmp_path / "tokens.toml"

    save_tokens(updated, output_path)
    reloaded = load_tokens(output_path)

    assert reloaded.global_colors.primary == "#223344"
    assert reloaded.palette.window == "#112233"
    assert reloaded.fonts.base_px == 15
    assert reloaded.metrics.corner_radius_px == 9


def test_visual_lab_token_updates_reject_invalid_colors():
    tokens = load_tokens()

    try:
        update_token_values(tokens, (("palette.window", "blue"),))
    except ValueError as exc:
        assert "palette.window" in str(exc)
    else:
        raise AssertionError("invalid color update should fail")


def test_visual_lab_scene_includes_required_layer_states():
    presentation = build_visual_lab_presentation()

    state = build_current_visual_lab_state()

    assert presentation.timeline_id == state.timeline.id
    assert str(DRUMS_LAYER_ID) in {str(layer.layer_id) for layer in presentation.layers}
    assert SELECTED_EVENT_ID in presentation.selected_event_ids
    assert not any(str(layer.layer_id).startswith("vl_") for layer in presentation.layers)
    assert any(
        layer.kind.value == "audio" and layer.parent_layer_id is None
        for layer in presentation.layers
    )
    assert any(layer.parent_layer_id is not None for layer in presentation.layers)
    assert any(layer.is_selected for layer in presentation.layers)
    assert any(layer.muted for layer in presentation.layers)
    assert any(layer.status.stale for layer in presentation.layers)
    assert presentation.selected_event_ids


def test_visual_lab_catalog_supports_folder_navigation_and_parts():
    registry = build_visual_lab_catalog()

    assert registry.categories() == (
        "Theme",
        "Timeline / rows",
        "Timeline / headers",
        "Timeline / canvas parts",
        "Chrome",
        "Panels",
        "Dialogs / forms",
        "Waveforms",
        "Primitives",
        "Compositions",
    )
    assert registry.first().entry_id == "theme.global-colors"
    assert registry.next_id("timeline.row.source-audio") == "timeline.row.stem-child"
    assert registry.previous_id("timeline.row.source-audio") == "theme.global-colors"
    assert registry.get("composition.timeline-current").part_ids
    assert registry.get("theme.global-colors").editable_token_paths


def test_visual_lab_catalog_entries_declare_source_truth():
    registry = build_visual_lab_catalog()
    allowed_source_kinds = {
        "production-backed",
        "current-model synthetic",
        "lab-only experimental",
    }

    for entry in registry.entries:
        assert entry.source_kind in allowed_source_kinds
        assert entry.source_path

    assert registry.get("timeline.row.source-audio").source_kind == "production-backed"
    assert "TimelineCanvas" in registry.get("timeline.row.source-audio").source_path
    assert registry.get("timeline.header.stale-cues").source_kind == "production-backed"
    assert "LayerHeaderBlock" in registry.get("timeline.header.stale-cues").source_path
    assert registry.get("primitive.control-card").source_kind == "lab-only experimental"
    assert registry.get("panel.object-info").source_kind == "production-backed"
    assert "ObjectInfoPanel" in registry.get("panel.object-info").source_path
    assert registry.get("waveform.event-preview.synthetic-fun").source_kind == (
        "current-model synthetic"
    )


def test_visual_lab_catalog_covers_current_ui_surface_categories():
    registry = build_visual_lab_catalog()
    required_entry_ids = {
        "chrome.editor-mode-bar",
        "chrome.transport-status",
        "panel.song-browser",
        "panel.object-info",
        "form.settings-page",
        "timeline.ruler",
        "timeline.waveform-preview-row",
        "waveform.event-preview.synthetic-fun",
        "composition.timeline-shell-current",
    }

    assert required_entry_ids <= {entry.entry_id for entry in registry.entries}
    assert "panel.object-info" in registry.get("composition.timeline-shell-current").part_ids


def test_visual_lab_catalog_filters_editable_tokens_by_selected_entry():
    tokens = load_tokens()
    registry = build_visual_lab_catalog()

    global_specs = registry.editable_specs_for("theme.global-colors", tokens)
    chip_specs = registry.editable_specs_for("primitive.status-chips", tokens)
    row_specs = registry.editable_specs_for("timeline.row.source-audio", tokens)

    assert {spec.section for spec in global_specs} == {"global_colors"}
    assert {spec.path for spec in chip_specs} >= {
        "palette.status_ok",
        "palette.status_sync",
        "palette.status_stale",
        "palette.status_muted",
    }
    assert "metrics.audio_row_height_px" not in {spec.path for spec in chip_specs}
    assert "metrics.audio_row_height_px" in {spec.path for spec in row_specs}


def test_visual_lab_catalog_supports_nested_style_targets():
    tokens = load_tokens()
    registry = build_visual_lab_catalog()

    targets = registry.editable_targets_for("timeline.row.source-audio", tokens)
    targets_by_id = {target.target_id: target for target in targets}

    assert "timeline.layer_row.header.title" in targets_by_id
    assert "timeline.layer_row.header.badge" in targets_by_id
    assert {spec.path for spec in targets_by_id["timeline.layer_row.header.title"].specs} >= {
        "palette.text",
        "palette.text_muted",
        "fonts.family",
        "fonts.label_px",
    }


def test_visual_lab_part_tree_filters_nested_target_knobs():
    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="timeline.row.source-audio")
    try:
        tree_item = _find_part_tree_item(
            preview.token_editor.part_tree,
            "timeline.layer_row.header.badge",
        )
        assert tree_item is not None

        preview.token_editor.part_tree.setCurrentItem(tree_item)

        visible_paths = set(preview.token_editor._editors)
        assert visible_paths == {
            "palette.status_ok",
            "palette.status_sync",
            "palette.status_stale",
            "palette.status_muted",
            "metrics.status_chip_height_px",
        }
    finally:
        preview.close()
        app.processEvents()


def test_visual_lab_live_editor_change_updates_tokens_and_refreshes_preview():
    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="primitive.status-chips")
    calls = []
    original_render_entry = preview._render_entry

    def record_render(entry, *, refresh_editor=True):
        calls.append((entry.entry_id, refresh_editor))
        original_render_entry(entry, refresh_editor=refresh_editor)

    preview._render_entry = record_render
    try:
        old_height = preview.tokens.metrics.status_chip_height_px
        editor = preview.token_editor._editors["metrics.status_chip_height_px"]
        assert isinstance(editor, QSpinBox)
        editor.setValue(old_height + 3)

        assert preview.tokens.metrics.status_chip_height_px == old_height + 3
        assert calls[-1] == ("primitive.status-chips", False)
    finally:
        preview.close()
        app.processEvents()


def _find_part_tree_item(tree, target_id: str):
    def visit(item):
        if item.data(0, Qt.ItemDataRole.UserRole) == target_id:
            return item
        for index in range(item.childCount()):
            found = visit(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(tree.topLevelItemCount()):
        found = visit(tree.topLevelItem(index))
        if found is not None:
            return found
    return None


def test_visual_lab_color_picker_path_can_be_monkeypatched():
    app = QApplication.instance() or QApplication([])
    spec = next(spec for spec in token_field_specs(load_tokens()) if spec.path == "palette.accent")
    editor = ColorTokenEditor(spec)
    try:
        editor.choose_color = lambda initial: QColor("#abcdef")

        editor.swatch.double_clicked.emit()

        assert editor.value() == "#abcdef"
    finally:
        editor.close()
        app.processEvents()


def test_visual_lab_font_family_editor_includes_missing_current_value():
    app = QApplication.instance() or QApplication([])
    spec = TokenFieldSpec(
        section="fonts",
        name="family",
        path="fonts.family",
        value_type=str,
        value="Missing Visual Lab Font",
    )
    editor = FontFamilyEditor(spec, font_families=("Installed Font",))
    try:
        assert editor.combo.findText("Missing Visual Lab Font") >= 0
        assert editor.value() == "Missing Visual Lab Font"
    finally:
        editor.close()
        app.processEvents()


def test_visual_lab_font_dropdown_change_live_updates_preview():
    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="primitive.status-chips")
    calls = []
    original_render_entry = preview._render_entry

    def record_render(entry, *, refresh_editor=True):
        calls.append((entry.entry_id, refresh_editor))
        original_render_entry(entry, refresh_editor=refresh_editor)

    preview._render_entry = record_render
    try:
        editor = preview.token_editor._editors["fonts.family"]
        assert isinstance(editor, FontFamilyEditor)

        editor.combo.setCurrentText("Visual Lab Test Font")

        assert preview.tokens.fonts.family == "Visual Lab Test Font"
        assert calls[-1] == ("primitive.status-chips", False)
    finally:
        preview.close()
        app.processEvents()


def test_visual_lab_font_import_helper_copies_supported_font_files(tmp_path):
    source_font = tmp_path / "DisplayTest.ttf"
    source_font.write_bytes(b"not a real font, copy-path test only")
    asset_dir = tmp_path / "assets" / "fonts"

    imported = import_lab_fonts([source_font], asset_dir=asset_dir, register=False)

    assert len(imported) == 1
    assert imported[0].asset_path == asset_dir / "DisplayTest.ttf"
    assert imported[0].asset_path.read_bytes() == source_font.read_bytes()
    assert imported[0].application_font_id is None
    assert list_lab_font_assets(asset_dir) == (asset_dir / "DisplayTest.ttf",)


def test_visual_lab_font_import_helper_rejects_unsupported_extensions(tmp_path):
    source_font = tmp_path / "DisplayTest.txt"
    source_font.write_text("not a font", encoding="utf-8")

    try:
        import_lab_fonts([source_font], asset_dir=tmp_path / "assets", register=False)
    except ValueError as exc:
        assert ".ttf" in str(exc)
        assert ".otf" in str(exc)
        assert ".ttc" in str(exc)
    else:
        raise AssertionError("unsupported font extension should fail")


def test_visual_lab_font_import_action_refreshes_visible_dropdowns(monkeypatch, tmp_path):
    import dev.visual_lab.token_editor as token_editor

    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="primitive.status-chips")
    source_font = tmp_path / "ImportedDisplay.otf"
    source_font.write_bytes(b"fake font data")

    class Imported:
        families = ("Imported Display",)

    monkeypatch.setattr(token_editor, "import_lab_fonts", lambda paths: (Imported(),))
    preview.token_editor.choose_font_files = lambda: (str(source_font),)
    try:
        editor = preview.token_editor._editors["fonts.family"]
        assert isinstance(editor, FontFamilyEditor)

        preview.token_editor.import_fonts()

        assert editor.combo.findText("Imported Display") >= 0
        assert "registered 1 family" in preview.token_editor.status_label.text()
    finally:
        preview.close()
        app.processEvents()


def test_visual_lab_fun_waveform_provider_registers_current_cache_shape():
    peaks = build_fun_waveform_peaks(64)
    cached = register_fun_waveform_preview()
    preview = build_fun_event_preview_state()

    assert peaks.shape == (64, 2)
    assert peaks.dtype.name == "float32"
    assert float(peaks[:, 1].max()) > 0.80
    assert float(peaks[:, 0].min()) < -0.45
    assert cached.peaks.shape[1] == 2
    assert get_cached_waveform(FUN_WAVEFORM_KEY) is not None
    assert preview.waveform_key == FUN_WAVEFORM_KEY
    assert preview.duration_seconds > 0.0


def test_visual_lab_catalog_preview_constructs_selected_item():
    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="primitive.status-chips")
    try:
        assert preview.selected_entry_id == "primitive.status-chips"
        assert preview.metadata.text().startswith(
            "Primitives / primitive / lab-only experimental"
        )
        assert "Source: dev.visual_lab.widgets lab-only primitives" in preview.metadata.text()
        assert preview.token_editor is not None
        preview.select_next()
        assert preview.selected_entry_id == "primitive.control-card"
    finally:
        preview.close()
        app.processEvents()


def test_visual_lab_catalog_entries_render_widgets():
    app = QApplication.instance() or QApplication([])
    tokens = load_tokens()
    registry = build_visual_lab_catalog()
    widgets = []
    try:
        for entry in registry.entries:
            widget = entry.render(tokens)
            widgets.append(widget)
            assert widget.minimumWidth() > 0
            assert entry.name
    finally:
        for widget in widgets:
            widget.close()
        app.processEvents()


def test_visual_lab_uses_current_component_imports_without_legacy_demo_modules():
    visual_lab_files = [
        path
        for path in Path("dev/visual_lab").glob("*.py")
        if path.name != "__init__.py"
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in visual_lab_files)

    assert "echozero.application.timeline.assembler" in source_text
    assert "echozero.ui.qt.timeline.widget_canvas" in source_text
    assert "echozero.ui.qt.timeline.blocks.layer_header" in source_text
    assert "echozero.ui.qt.timeline.demo_app" not in source_text
    assert "echozero.ui.qt.timeline.fixture_loader" not in source_text
    assert "load_realistic_timeline_fixture" not in source_text


def test_visual_lab_qt_grab_fallback_captures_png(tmp_path):
    app = QApplication.instance() or QApplication([])
    preview = build_preview(selected_entry_id="timeline.header.stale-cues")
    output_path = tmp_path / "visual_lab.png"
    try:
        preview.show()
        app.processEvents()

        result = capture_widget(preview, output_path, prefer_peekaboo=False)

        assert result.backend == "qt-grab"
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        preview.close()
        app.processEvents()
