"""Visual Lab catalog source labels.
Exists to keep entry factories focused on registration shape, not metadata text.
Used by support-only catalog entries and tests.
"""

from __future__ import annotations

TIMELINE_CANVAS_SOURCE = (
    "echozero.ui.qt.timeline.widget_canvas.TimelineCanvas with "
    "echozero.application.timeline.assembler.TimelineAssembler"
)
LAYER_HEADER_SOURCE = "echozero.ui.qt.timeline.blocks.layer_header.LayerHeaderBlock"
TRANSPORT_SOURCE = "echozero.ui.qt.timeline.widget_controls.TransportBar"
EDITOR_BAR_SOURCE = "echozero.ui.qt.timeline.widget_controls.TimelineEditorModeBar"
RULER_SOURCE = "echozero.ui.qt.timeline.widget_controls.TimelineRuler"
TIMELINE_SHELL_SOURCE = "echozero.ui.qt.timeline.widget.TimelineWidget"
SONG_BROWSER_SOURCE = "echozero.ui.qt.song_browser_panel.SongBrowserPanel"
OBJECT_INFO_SOURCE = "echozero.ui.qt.timeline.object_info_panel.ObjectInfoPanel"
SETTINGS_FORM_SOURCE = "echozero.ui.qt.settings_page_form.SettingsPageForm"
WAVEFORM_PREVIEW_SOURCE = (
    "echozero.ui.qt.timeline.object_info_panel_preview.EventPreviewWaveform + "
    "dev.visual_lab.waveforms"
)
LAB_PRIMITIVE_SOURCE = "dev.visual_lab.widgets lab-only primitives"
