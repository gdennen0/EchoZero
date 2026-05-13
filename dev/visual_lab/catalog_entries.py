"""Visual Lab default catalog entries.
Exists to keep the runner small while making object registration explicit.
Each factory returns an isolated primitive, element, chrome object, or composition.
"""

from __future__ import annotations

from dev.visual_lab.catalog import CatalogEntry, CatalogRegistry, flatten_categories
from dev.visual_lab.current_state import (
    BASS_LAYER_ID, CUES_LAYER_ID, DRUMS_LAYER_ID, SOURCE_AUDIO_LAYER_ID, VOCALS_LAYER_ID,
    build_current_visual_lab_presentation, current_layer_by_id,
)
from dev.visual_lab.catalog_sources import (
    EDITOR_BAR_SOURCE, LAB_PRIMITIVE_SOURCE, LAYER_HEADER_SOURCE, OBJECT_INFO_SOURCE,
    RULER_SOURCE, SETTINGS_FORM_SOURCE, SONG_BROWSER_SOURCE, TIMELINE_CANVAS_SOURCE,
    TIMELINE_SHELL_SOURCE, TRANSPORT_SOURCE, WAVEFORM_PREVIEW_SOURCE,
)
from dev.visual_lab.catalog_theme import theme_entries
from dev.visual_lab.editable_tokens import SHELL_TOKENS
from dev.visual_lab.style_targets import (
    CONTROL_TARGETS, STATUS_CHIP_TARGETS, TIMELINE_HEADER_TARGETS, TIMELINE_LAYER_ROW_TARGETS,
    TRANSPORT_TARGETS, WAVEFORM_TARGETS,
)
from dev.visual_lab.tokens import load_tokens
from dev.visual_lab.widgets import (
    CatalogFrame, ControlPrimitivePreviewWidget, EditorModeBarPreviewWidget,
    LayerHeaderPreviewWidget, ObjectInfoPanelPreviewWidget, SettingsFormPreviewWidget,
    SongBrowserPanelPreviewWidget, StatusChipPreviewWidget, TimelineCanvasPreviewWidget,
    TimelineRulerPreviewWidget, TimelineShellPreviewWidget, TransportBarPreviewWidget,
    WaveformPreviewWidget,
)
from echozero.application.presentation.models import TimelinePresentation


def build_visual_lab_catalog() -> CatalogRegistry:
    """Build the default Visual Lab catalog registry."""
    presentation = build_current_visual_lab_presentation()
    entries = flatten_categories(
        (
            ("Theme", theme_entries()),
            ("Timeline / rows", _timeline_row_entries(presentation)),
            ("Timeline / headers", _timeline_header_entries()),
            ("Timeline / canvas parts", _timeline_canvas_part_entries(presentation)),
            ("Chrome", _chrome_entries(presentation)),
            ("Panels", _panel_entries(presentation)),
            ("Dialogs / forms", _dialog_form_entries()),
            ("Waveforms", _waveform_entries()),
            ("Primitives", _primitive_entries()),
            ("Compositions", _composition_entries(presentation)),
        )
    )
    registry = CatalogRegistry(entries)
    registry.validate_parts()
    registry.validate_editable_tokens(load_tokens())
    return registry


def _timeline_row_entries(presentation: TimelinePresentation) -> tuple[CatalogEntry, ...]:
    return (
        _timeline_entry(
            presentation,
            "timeline.row.source-audio",
            "Source audio row",
            "Production timeline canvas row assembled from current app timeline models.",
            (str(SOURCE_AUDIO_LAYER_ID),),
        ),
        _timeline_entry(
            presentation,
            "timeline.row.stem-child",
            "Stem child row",
            "Production timeline canvas row for a current-model nested stem layer.",
            (str(VOCALS_LAYER_ID),),
        ),
        _timeline_entry(
            presentation,
            "timeline.row.stem-muted",
            "Muted stem row",
            "Production timeline canvas row for a muted current-model stem layer.",
            (str(BASS_LAYER_ID),),
        ),
        _timeline_entry(
            presentation,
            "timeline.row.selected-events",
            "Selected event row",
            "Production timeline canvas row with current selection semantics.",
            (str(DRUMS_LAYER_ID),),
        ),
        _timeline_entry(
            presentation,
            "timeline.row.stale-cues",
            "Stale cue row",
            "Production timeline canvas row with current stale status semantics.",
            (str(CUES_LAYER_ID),),
        ),
    )


def _timeline_header_entries() -> tuple[CatalogEntry, ...]:
    return (
        _header_entry(
            "timeline.header.source-audio",
            "Source audio header",
            str(SOURCE_AUDIO_LAYER_ID),
        ),
        _header_entry("timeline.header.stem-child", "Stem child header", str(VOCALS_LAYER_ID)),
        _header_entry("timeline.header.stem-muted", "Muted stem header", str(BASS_LAYER_ID)),
        _header_entry(
            "timeline.header.selected-events",
            "Selected event header",
            str(DRUMS_LAYER_ID),
        ),
        _header_entry("timeline.header.stale-cues", "Stale cue header", str(CUES_LAYER_ID)),
    )


def _chrome_entries(presentation: TimelinePresentation) -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="chrome.editor-mode-bar",
            name="Header/editor toolbar",
            category="Chrome",
            description=(
                "Production top timeline toolbar; this is the closest current top chrome "
                "component to an app header bar."
            ),
            kind="chrome",
            source_kind="production-backed",
            source_path=EDITOR_BAR_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Header/editor toolbar",
                EditorModeBarPreviewWidget(tokens),
                width=980,
                height=150,
            ),
            part_ids=("primitive.control-card",),
            style_targets=CONTROL_TARGETS,
        ),
        CatalogEntry(
            entry_id="chrome.transport-status",
            name="Transport/status chrome",
            category="Chrome",
            description=(
                "Production transport bar rendered with current assembled presentation state."
            ),
            kind="composite",
            source_kind="production-backed",
            source_path=TRANSPORT_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Transport/status chrome",
                TransportBarPreviewWidget(tokens, presentation),
                height=180,
            ),
            part_ids=("primitive.control-card", "primitive.status-chips"),
            style_targets=TRANSPORT_TARGETS + CONTROL_TARGETS + STATUS_CHIP_TARGETS,
        ),
    )


def _timeline_canvas_part_entries(
    presentation: TimelinePresentation,
) -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="timeline.ruler",
            name="Timeline ruler",
            category="Timeline / canvas parts",
            description="Production ruler chrome rendered against current timeline scale.",
            kind="element",
            source_kind="production-backed",
            source_path=RULER_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Timeline ruler",
                TimelineRulerPreviewWidget(tokens, presentation),
                width=920,
                height=150,
            ),
            editable_token_paths=(
                "palette.panel",
                "palette.border",
                "palette.grid",
                "palette.text_muted",
                "fonts.family",
                "fonts.small_px",
                "metrics.timeline_width_px",
            ),
        ),
        CatalogEntry(
            entry_id="timeline.waveform-preview-row",
            name="Fun waveform preview row",
            category="Timeline / canvas parts",
            description=(
                "Current-model audio row preview with synthetic sine/pulse waveform cache data."
            ),
            kind="element",
            source_kind="current-model synthetic",
            source_path=WAVEFORM_PREVIEW_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Fun waveform preview row",
                TimelineCanvasPreviewWidget(
                    tokens,
                    _presentation_for_layers(presentation, (str(SOURCE_AUDIO_LAYER_ID),)),
                ),
                width=920,
                height=260,
            ),
            part_ids=("timeline.row.source-audio", "waveform.event-preview.synthetic-fun"),
            style_targets=TIMELINE_LAYER_ROW_TARGETS + WAVEFORM_TARGETS,
        ),
    )


def _panel_entries(presentation: TimelinePresentation) -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="panel.song-browser",
            name="Setlist browser panel",
            category="Panels",
            description="Production left-side song browser populated from current model options.",
            kind="element",
            source_kind="production-backed",
            source_path=SONG_BROWSER_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Setlist browser panel",
                SongBrowserPanelPreviewWidget(tokens, presentation),
                width=420,
                height=660,
            ),
            editable_token_paths=SHELL_TOKENS,
        ),
        CatalogEntry(
            entry_id="panel.object-info",
            name="Object info palette",
            category="Panels",
            description=(
                "Production inspector/object info panel rendered from the current selected "
                "timeline object contract."
            ),
            kind="element",
            source_kind="production-backed",
            source_path=OBJECT_INFO_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Object info palette",
                ObjectInfoPanelPreviewWidget(tokens, presentation),
                width=460,
                height=700,
            ),
            part_ids=("waveform.event-preview.synthetic-fun", "primitive.control-card"),
            style_targets=CONTROL_TARGETS + WAVEFORM_TARGETS,
        ),
    )


def _dialog_form_entries() -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="form.settings-page",
            name="Settings page form",
            category="Dialogs / forms",
            description=(
                "Production reusable settings form rendered with a small current-shape "
                "application settings page."
            ),
            kind="element",
            source_kind="production-backed",
            source_path=SETTINGS_FORM_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Settings page form",
                SettingsFormPreviewWidget(tokens),
                width=720,
                height=660,
            ),
            style_targets=CONTROL_TARGETS,
        ),
    )


def _waveform_entries() -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="waveform.event-preview.synthetic-fun",
            name="Fun event waveform",
            category="Waveforms",
            description=(
                "Inspector waveform widget using a current-shape synthetic sine/pulse cache "
                "provider for readable preview states."
            ),
            kind="element",
            source_kind="current-model synthetic",
            source_path=WAVEFORM_PREVIEW_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Fun event waveform",
                WaveformPreviewWidget(tokens),
                width=620,
                height=190,
            ),
            style_targets=WAVEFORM_TARGETS,
        ),
    )


def _primitive_entries() -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="primitive.status-chips",
            name="Status chips",
            category="Primitives",
            description="Lab-only chip vocabulary retained as token exploration.",
            kind="primitive",
            source_kind="lab-only experimental",
            source_path=LAB_PRIMITIVE_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Status chips",
                StatusChipPreviewWidget(
                    tokens,
                    (
                        ("ready", tokens.palette.status_ok),
                        ("sync", tokens.palette.status_sync),
                        ("stale", tokens.palette.status_stale),
                        ("muted", tokens.palette.status_muted),
                    ),
                ),
                height=180,
            ),
            style_targets=STATUS_CHIP_TARGETS,
        ),
        CatalogEntry(
            entry_id="primitive.control-card",
            name="Panel/card/buttons",
            category="Primitives",
            description="Lab-only wrapper for basic panel, card, and button-like primitives.",
            kind="primitive",
            source_kind="lab-only experimental",
            source_path=LAB_PRIMITIVE_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Panel/card/buttons",
                ControlPrimitivePreviewWidget(tokens),
                height=240,
            ),
            style_targets=CONTROL_TARGETS,
        ),
    )


def _composition_entries(presentation: TimelinePresentation) -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            entry_id="composition.timeline-current",
            name="Current timeline canvas",
            category="Compositions",
            description=(
                "Production timeline canvas assembled from current app timeline/session models."
            ),
            kind="composition",
            source_kind="production-backed",
            source_path=TIMELINE_CANVAS_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Current timeline canvas",
                TimelineCanvasPreviewWidget(tokens, presentation),
                width=920,
                height=520,
            ),
            part_ids=(
                "timeline.row.source-audio",
                "timeline.row.stem-child",
                "timeline.row.stem-muted",
                "timeline.row.selected-events",
                "timeline.row.stale-cues",
            ),
            style_targets=TIMELINE_LAYER_ROW_TARGETS,
        ),
        CatalogEntry(
            entry_id="composition.timeline-shell-current",
            name="Current timeline shell",
            category="Compositions",
            description=(
                "Production timeline shell composition including setlist browser, editor "
                "toolbar, ruler, canvas, inspector palette, scrollbar, and transport chrome."
            ),
            kind="composition",
            source_kind="production-backed",
            source_path=TIMELINE_SHELL_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Current timeline shell",
                TimelineShellPreviewWidget(tokens, presentation),
                width=1220,
                height=820,
            ),
            part_ids=(
                "panel.song-browser",
                "chrome.editor-mode-bar",
                "timeline.ruler",
                "composition.timeline-current",
                "panel.object-info",
                "chrome.transport-status",
            ),
            style_targets=(
                CONTROL_TARGETS + TIMELINE_LAYER_ROW_TARGETS + WAVEFORM_TARGETS
                + TRANSPORT_TARGETS
            ),
        ),
    )


def _timeline_entry(
    presentation: TimelinePresentation,
    entry_id: str,
    name: str,
    description: str,
    layer_ids: tuple[str, ...],
) -> CatalogEntry:
    return CatalogEntry(
        entry_id=entry_id,
        name=name,
        category="Timeline / rows",
        description=description,
        kind="element",
        source_kind="production-backed",
        source_path=TIMELINE_CANVAS_SOURCE,
        render=lambda tokens: CatalogFrame(
            tokens,
            name,
            TimelineCanvasPreviewWidget(tokens, _presentation_for_layers(presentation, layer_ids)),
            height=360,
        ),
        part_ids=tuple(_header_part_id(layer_id) for layer_id in layer_ids),
        style_targets=TIMELINE_LAYER_ROW_TARGETS,
    )


def _header_entry(entry_id: str, name: str, layer_id: str) -> CatalogEntry:
    return CatalogEntry(
        entry_id=entry_id,
        name=name,
        category="Timeline / headers",
        description="Production layer header paint block rendered as an independent object.",
        kind="element",
        source_kind="production-backed",
        source_path=LAYER_HEADER_SOURCE,
        render=lambda tokens: CatalogFrame(
            tokens,
            name,
            LayerHeaderPreviewWidget(
                tokens,
                current_layer_by_id(layer_id),
                has_child_layers=layer_id == str(SOURCE_AUDIO_LAYER_ID),
            ),
            width=460,
            height=180,
        ),
        style_targets=TIMELINE_HEADER_TARGETS,
    )


def _header_part_id(layer_id: str) -> str:
    mapping = {
        str(SOURCE_AUDIO_LAYER_ID): "timeline.header.source-audio",
        str(VOCALS_LAYER_ID): "timeline.header.stem-child",
        str(BASS_LAYER_ID): "timeline.header.stem-muted",
        str(DRUMS_LAYER_ID): "timeline.header.selected-events",
        str(CUES_LAYER_ID): "timeline.header.stale-cues",
    }
    return mapping[layer_id]


def _presentation_for_layers(
    presentation: TimelinePresentation, layer_ids: tuple[str, ...]
) -> TimelinePresentation:
    layers = [current_layer_by_id(layer_id) for layer_id in layer_ids]
    return TimelinePresentation(
        timeline_id=presentation.timeline_id,
        title=presentation.title,
        active_song_title=presentation.active_song_title,
        active_song_version_label=presentation.active_song_version_label,
        layers=layers,
        playhead=presentation.playhead,
        follow_mode=presentation.follow_mode,
        selected_layer_id=presentation.selected_layer_id,
        selected_event_ids=list(presentation.selected_event_ids),
        pixels_per_second=presentation.pixels_per_second,
        current_time_label=presentation.current_time_label,
        end_time_label=presentation.end_time_label,
    )
