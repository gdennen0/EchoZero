"""Timeline shell style tokens for Stage Zero Qt surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoxInsets:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class ObjectPaletteStyle:
    frame_object_name: str
    title_object_name: str
    section_object_name: str
    kind_object_name: str
    body_object_name: str
    background_hex: str
    border_hex: str
    title_hex: str
    section_hex: str
    kind_fg_hex: str
    kind_bg_hex: str
    kind_border_hex: str
    body_hex: str
    button_bg_hex: str
    button_fg_hex: str
    button_border_hex: str
    button_disabled_fg_hex: str
    button_disabled_bg_hex: str
    button_disabled_border_hex: str
    spin_fg_hex: str
    spin_bg_hex: str
    spin_border_hex: str
    title_font_px: int
    section_font_px: int
    body_font_px: int
    kind_font_px: int
    title_font_weight: int
    section_font_weight: int
    kind_font_weight: int
    button_font_weight: int
    button_min_height_px: int
    spin_min_height_px: int
    min_width_px: int
    max_width_px: int
    content_padding: BoxInsets
    section_spacing_px: int


@dataclass(frozen=True, slots=True)
class TimelineShellStyle:
    canvas_background_hex: str
    scroll_area_background_hex: str
    object_palette: ObjectPaletteStyle


TIMELINE_STYLE = TimelineShellStyle(
    canvas_background_hex="#12151b",
    scroll_area_background_hex="#12151b",
    object_palette=ObjectPaletteStyle(
        frame_object_name="timeline_object_info",
        title_object_name="timeline_object_info_title",
        section_object_name="timeline_object_info_section",
        kind_object_name="timeline_object_info_kind",
        body_object_name="timeline_object_info_body",
        background_hex="#171d26",
        border_hex="#252c38",
        title_hex="#f0f3f8",
        section_hex="#8da1b9",
        kind_fg_hex="#d6e3f6",
        kind_bg_hex="#223347",
        kind_border_hex="#35506d",
        body_hex="#c2cad6",
        button_bg_hex="#223041",
        button_fg_hex="#dbe8f8",
        button_border_hex="#32455d",
        button_disabled_fg_hex="#6b7481",
        button_disabled_bg_hex="#1a212b",
        button_disabled_border_hex="#263142",
        spin_fg_hex="#dce8fb",
        spin_bg_hex="#1e2a38",
        spin_border_hex="#344a65",
        title_font_px=12,
        section_font_px=9,
        body_font_px=10,
        kind_font_px=10,
        title_font_weight=700,
        section_font_weight=700,
        kind_font_weight=600,
        button_font_weight=600,
        button_min_height_px=26,
        spin_min_height_px=24,
        min_width_px=300,
        max_width_px=500,
        content_padding=BoxInsets(left=14, top=12, right=14, bottom=12),
        section_spacing_px=6,
    ),
)


def build_object_palette_stylesheet(style: ObjectPaletteStyle = TIMELINE_STYLE.object_palette) -> str:
    return f"""
        QFrame#{style.frame_object_name} {{
            background: {style.background_hex};
            border-left: 1px solid {style.border_hex};
        }}
        QLabel#{style.title_object_name} {{
            color: {style.title_hex};
            font-size: {style.title_font_px}px;
            font-weight: {style.title_font_weight};
        }}
        QLabel#{style.section_object_name} {{
            color: {style.section_hex};
            font-size: {style.section_font_px}px;
            font-weight: {style.section_font_weight};
        }}
        QLabel#{style.kind_object_name} {{
            color: {style.kind_fg_hex};
            background: {style.kind_bg_hex};
            border: 1px solid {style.kind_border_hex};
            border-radius: 10px;
            padding: 2px 8px;
            font-size: {style.kind_font_px}px;
            font-weight: {style.kind_font_weight};
        }}
        QLabel#{style.body_object_name} {{
            color: {style.body_hex};
            font-size: {style.body_font_px}px;
        }}
        QPushButton {{
            background: {style.button_bg_hex};
            color: {style.button_fg_hex};
            border: 1px solid {style.button_border_hex};
            border-radius: 4px;
            padding: 5px 8px;
            min-height: {style.button_min_height_px}px;
            font-size: 10px;
            font-weight: {style.button_font_weight};
        }}
        QPushButton:disabled {{
            color: {style.button_disabled_fg_hex};
            background: {style.button_disabled_bg_hex};
            border-color: {style.button_disabled_border_hex};
        }}
        QDoubleSpinBox {{
            color: {style.spin_fg_hex};
            background: {style.spin_bg_hex};
            border: 1px solid {style.spin_border_hex};
            border-radius: 4px;
            padding: 3px 6px;
            min-height: {style.spin_min_height_px}px;
        }}
    """


def build_timeline_scroll_area_stylesheet(style: TimelineShellStyle = TIMELINE_STYLE) -> str:
    return f"background: {style.scroll_area_background_hex}; border: none;"
