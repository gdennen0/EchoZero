"""Timeline shell style tokens for Stage Zero Qt surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BoxInsets:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class FontStyle:
    point_size: int
    bold: bool = False
    weight: int | None = None


@dataclass(frozen=True, slots=True)
class PaintButtonStyle:
    fill_hex: str
    border_hex: str
    text_hex: str
    corner_radius: int
    font: FontStyle


@dataclass(frozen=True, slots=True)
class ToggleButtonStateStyle:
    fill_hex: str
    text_hex: str


@dataclass(frozen=True, slots=True)
class MuteSoloButtonStyle:
    active: ToggleButtonStateStyle
    inactive: ToggleButtonStateStyle
    dimmed_inactive_fill_hex: str
    border_hex: str
    corner_radius: int
    font: FontStyle


@dataclass(frozen=True, slots=True)
class StatusChipStyle:
    fill_hex: str
    text_hex: str
    corner_radius: int
    font: FontStyle


@dataclass(frozen=True, slots=True)
class LayerHeaderStatusStyles:
    stale: StatusChipStyle
    edited: StatusChipStyle


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
class TimelineCanvasStyle:
    background_hex: str
    row_fill_hex: str
    selected_row_fill_hex: str
    dimmed_row_fill_hex: str
    row_divider_hex: str
    no_takes_hint_hex: str
    no_takes_hint_dimmed_hex: str
    region_even_hex: str
    region_odd_hex: str
    region_alpha: int
    region_selected_outline_hex: str


@dataclass(frozen=True, slots=True)
class TimelinePlayheadStyle:
    color_hex: str
    line_width_px: int
    head_outline_width_px: int


@dataclass(frozen=True, slots=True)
class TransportBarStyle:
    background_hex: str
    title_hex: str
    time_hex: str
    meta_hex: str
    button: PaintButtonStyle


@dataclass(frozen=True, slots=True)
class LayerHeaderStyle:
    background_hex: str
    selected_background_hex: str
    dimmed_background_hex: str
    title_hex: str
    dimmed_title_hex: str
    title_font: FontStyle
    toggle_fill_hex: str
    toggle_border_hex: str
    toggle_text_hex: str
    toggle_corner_radius: int
    toggle_font: FontStyle
    status: LayerHeaderStatusStyles
    mute_solo: MuteSoloButtonStyle


@dataclass(frozen=True, slots=True)
class TakeActionChipStyle:
    fill_hex: str
    text_hex: str
    font: FontStyle


@dataclass(frozen=True, slots=True)
class TakeRowStyle:
    row_fill_hex: str
    selected_row_fill_hex: str
    dimmed_row_fill_hex: str
    header_fill_hex: str
    selected_header_fill_hex: str
    dimmed_header_fill_hex: str
    divider_hex: str
    label_hex: str
    dimmed_label_hex: str
    options_button_open_fill_hex: str
    options_button_closed_fill_hex: str
    options_button_dimmed_fill_hex: str
    options_button_open_text_hex: str
    options_button_closed_text_hex: str
    options_button_font: FontStyle
    options_area_fill_hex: str
    action_chip: TakeActionChipStyle


@dataclass(frozen=True, slots=True)
class EventLaneStyle:
    default_fill_hex: str
    dimmed_alpha: int
    selection_lighten_factor: int
    border_darkness_factor: int
    normal_border_width_px: int
    selected_border_width_px: int
    text_hex: str
    corner_radius: int


@dataclass(frozen=True, slots=True)
class WaveformLaneStyle:
    dimmed_alpha: int
    fallback_pen_width_px: float
    cached_pen_width_px: float
    fallback_amp_row_factor: float
    cached_amp_row_factor: float


@dataclass(frozen=True, slots=True)
class RulerStyle:
    background_hex: str
    divider_hex: str
    header_background_hex: str
    title_hex: str
    tick_hex: str
    grid_hex: str
    label_hex: str
    region_even_hex: str
    region_odd_hex: str
    region_alpha: int
    region_border_hex: str
    region_label_hex: str


@dataclass(frozen=True, slots=True)
class TimelineFixtureStyle:
    default_sync_label: str
    fallback_audio_lane_hex: str
    layer_color_tokens: MappingProxyType
    take_action_labels: MappingProxyType


@dataclass(frozen=True, slots=True)
class TimelineShellStyle:
    window_title: str
    canvas: TimelineCanvasStyle
    playhead: TimelinePlayheadStyle
    scroll_area_background_hex: str
    object_palette: ObjectPaletteStyle
    transport_bar: TransportBarStyle
    layer_header: LayerHeaderStyle
    take_row: TakeRowStyle
    event_lane: EventLaneStyle
    waveform_lane: WaveformLaneStyle
    ruler: RulerStyle
    fixture: TimelineFixtureStyle


TIMELINE_STYLE = TimelineShellStyle(
    window_title="EchoZero Timeline Preview",
    canvas=TimelineCanvasStyle(
        background_hex="#12151b",
        row_fill_hex="#161b22",
        selected_row_fill_hex="#1a212b",
        dimmed_row_fill_hex="#12161c",
        row_divider_hex="#252c38",
        no_takes_hint_hex="#8b97a8",
        no_takes_hint_dimmed_hex="#5f6977",
        region_even_hex="#d9dee6",
        region_odd_hex="#c9d0da",
        region_alpha=16,
        region_selected_outline_hex="#9fc3ff",
    ),
    playhead=TimelinePlayheadStyle(
        color_hex="#ff5f57",
        line_width_px=2,
        head_outline_width_px=1,
    ),
    scroll_area_background_hex="#12151b",
    object_palette=ObjectPaletteStyle(
        frame_object_name="objectInfoPanel",
        title_object_name="objectPaletteHeader",
        section_object_name="timeline_object_info_section",
        kind_object_name="timeline_object_info_kind",
        body_object_name="selectionSecondaryLabel",
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
        section_spacing_px=10,
    ),
    transport_bar=TransportBarStyle(
        background_hex="#0e1217",
        title_hex="#f0f3f8",
        time_hex="#f6f8fb",
        meta_hex="#93a0b1",
        button=PaintButtonStyle(
            fill_hex="#1b2330",
            border_hex="#334055",
            text_hex="#ffffff",
            corner_radius=6,
            font=FontStyle(point_size=9, bold=True),
        ),
    ),
    layer_header=LayerHeaderStyle(
        background_hex="#1b212a",
        selected_background_hex="#202833",
        dimmed_background_hex="#151922",
        title_hex="#f0f3f8",
        dimmed_title_hex="#cbd3df",
        title_font=FontStyle(point_size=10, bold=True),
        toggle_fill_hex="#141922",
        toggle_border_hex="#445065",
        toggle_text_hex="#d7dce4",
        toggle_corner_radius=6,
        toggle_font=FontStyle(point_size=9, bold=True),
        status=LayerHeaderStatusStyles(
            stale=StatusChipStyle(
                fill_hex="#7a5b16",
                text_hex="#f8c555",
                corner_radius=5,
                font=FontStyle(point_size=8, bold=True),
            ),
            edited=StatusChipStyle(
                fill_hex="#184c39",
                text_hex="#7fd1ae",
                corner_radius=5,
                font=FontStyle(point_size=8, bold=True),
            ),
        ),
        mute_solo=MuteSoloButtonStyle(
            active=ToggleButtonStateStyle(fill_hex="#2b6bf0", text_hex="#ffffff"),
            inactive=ToggleButtonStateStyle(fill_hex="#18202a", text_hex="#b8c0cc"),
            dimmed_inactive_fill_hex="#10151b",
            border_hex="#4b5669",
            corner_radius=5,
            font=FontStyle(point_size=8, bold=True),
        ),
    ),
    take_row=TakeRowStyle(
        row_fill_hex="#121821",
        selected_row_fill_hex="#1a212b",
        dimmed_row_fill_hex="#0f141b",
        header_fill_hex="#171d26",
        selected_header_fill_hex="#202833",
        dimmed_header_fill_hex="#141922",
        divider_hex="#222936",
        label_hex="#aeb8c6",
        dimmed_label_hex="#8e98a6",
        options_button_open_fill_hex="#263244",
        options_button_closed_fill_hex="#1f2938",
        options_button_dimmed_fill_hex="#1a2230",
        options_button_open_text_hex="#9fcbff",
        options_button_closed_text_hex="#8ea4bf",
        options_button_font=FontStyle(point_size=8, bold=True),
        options_area_fill_hex="#101822",
        action_chip=TakeActionChipStyle(
            fill_hex="#22364f",
            text_hex="#d0e4ff",
            font=FontStyle(point_size=8),
        ),
    ),
    event_lane=EventLaneStyle(
        default_fill_hex="#57a0ff",
        dimmed_alpha=120,
        selection_lighten_factor=130,
        border_darkness_factor=160,
        normal_border_width_px=1,
        selected_border_width_px=2,
        text_hex="#0b1220",
        corner_radius=5,
    ),
    waveform_lane=WaveformLaneStyle(
        dimmed_alpha=120,
        fallback_pen_width_px=1.2,
        cached_pen_width_px=1.0,
        fallback_amp_row_factor=0.30,
        cached_amp_row_factor=0.38,
    ),
    ruler=RulerStyle(
        background_hex="#0f1318",
        divider_hex="#2a303c",
        header_background_hex="#171c23",
        title_hex="#9aa4b2",
        tick_hex="#3b4352",
        grid_hex="#b8c0cc",
        label_hex="#b8c0cc",
        region_even_hex="#d9dee6",
        region_odd_hex="#c9d0da",
        region_alpha=28,
        region_border_hex="#9ea9b7",
        region_label_hex="#d7dee8",
    ),
    fixture=TimelineFixtureStyle(
        default_sync_label="No sync",
        fallback_audio_lane_hex="#9b87f5",
        layer_color_tokens=MappingProxyType(
            {
                "song": "#4da3ff",
                "drums": "#9b87f5",
                "bass": "#d68cff",
                "vocals": "#7dd3fc",
                "other": "#94a3b8",
                "kick": "#66a3ff",
                "snare": "#7fd1ae",
                "hihat": "#f8c555",
                "clap": "#ff8c78",
                "sync": "#ff8c78",
                "event_preview": "#7fd1ae",
            }
        ),
        take_action_labels=MappingProxyType(
            {
                "overwrite_main": "Overwrite Main",
                "merge_main": "Merge Main",
                "promote_take": "Promote Take",
                "delete_take": "Delete Take",
            }
        ),
    ),
)


def fixture_color(token: str, style: TimelineShellStyle = TIMELINE_STYLE) -> str:
    return style.fixture.layer_color_tokens[token]


def fixture_take_action_label(action_id: str, style: TimelineShellStyle = TIMELINE_STYLE) -> str:
    return style.fixture.take_action_labels[action_id]


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
        QPlainTextEdit#{style.body_object_name} {{
            color: {style.body_hex};
            background: transparent;
            border: none;
            font-size: {style.body_font_px}px;
            padding: 0;
        }}
        QSplitter#timeline_object_info_splitter::handle:vertical {{
            background: {style.border_hex};
            border-radius: 3px;
            margin: 1px 120px;
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
