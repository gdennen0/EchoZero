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
    split_divider_hex: str
    no_takes_hint_hex: str
    no_takes_hint_dimmed_hex: str
    section_even_hex: str
    section_odd_hex: str
    section_alpha: int
    section_boundary_hex: str
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
    demoted_fill_hex: str
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
    split_divider_hex: str
    header_background_hex: str
    title_hex: str
    tick_hex: str
    grid_hex: str
    label_hex: str
    section_even_hex: str
    section_odd_hex: str
    section_alpha: int
    section_boundary_hex: str
    section_label_hex: str
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
        background_hex="#101010",
        row_fill_hex="#171719",
        selected_row_fill_hex="#202022",
        dimmed_row_fill_hex="#121214",
        row_divider_hex="#3a383a",
        split_divider_hex="#5a565a",
        no_takes_hint_hex="#aaa49e",
        no_takes_hint_dimmed_hex="#685f67",
        section_even_hex="#885A2D",
        section_odd_hex="#8f8a84",
        section_alpha=22,
        section_boundary_hex="#d8d2cb",
        region_even_hex="#28262a",
        region_odd_hex="#242224",
        region_alpha=16,
        region_selected_outline_hex="#d8d2cb",
    ),
    playhead=TimelinePlayheadStyle(
        color_hex="#CC8844",
        line_width_px=1,
        head_outline_width_px=1,
    ),
    scroll_area_background_hex="#101010",
    object_palette=ObjectPaletteStyle(
        frame_object_name="objectInfoPanel",
        title_object_name="objectPaletteHeader",
        section_object_name="timeline_object_info_section",
        kind_object_name="timeline_object_info_kind",
        body_object_name="selectionSecondaryLabel",
        background_hex="#171719",
        border_hex="#3a383a",
        title_hex="#f6f3ee",
        section_hex="#aaa49e",
        kind_fg_hex="#f6f3ee",
        kind_bg_hex="#202022",
        kind_border_hex="#8f8a84",
        body_hex="#aaa49e",
        button_bg_hex="#202022",
        button_fg_hex="#e8e2dc",
        button_border_hex="#4a4749",
        button_disabled_fg_hex="#685f67",
        button_disabled_bg_hex="#18181a",
        button_disabled_border_hex="#3a383a",
        spin_fg_hex="#e8e2dc",
        spin_bg_hex="#202022",
        spin_border_hex="#4a4749",
        title_font_px=12,
        section_font_px=9,
        body_font_px=10,
        kind_font_px=10,
        title_font_weight=700,
        section_font_weight=700,
        kind_font_weight=600,
        button_font_weight=600,
        button_min_height_px=22,
        spin_min_height_px=22,
        min_width_px=244,
        max_width_px=360,
        content_padding=BoxInsets(left=6, top=6, right=6, bottom=6),
        section_spacing_px=4,
    ),
    transport_bar=TransportBarStyle(
        background_hex="#101010",
        title_hex="#e8e2dc",
        time_hex="#f6f3ee",
        meta_hex="#aaa49e",
        button=PaintButtonStyle(
            fill_hex="#202022",
            border_hex="#4a4749",
            text_hex="#f6f3ee",
            corner_radius=3,
            font=FontStyle(point_size=9, bold=True),
        ),
    ),
    layer_header=LayerHeaderStyle(
        background_hex="#18181a",
        selected_background_hex="#202022",
        dimmed_background_hex="#121214",
        title_hex="#f6f3ee",
        dimmed_title_hex="#807a74",
        title_font=FontStyle(point_size=10, bold=True),
        toggle_fill_hex="#18181a",
        toggle_border_hex="#4a4749",
        toggle_text_hex="#d8d2cb",
        toggle_corner_radius=3,
        toggle_font=FontStyle(point_size=9, bold=True),
        status=LayerHeaderStatusStyles(
            stale=StatusChipStyle(
                fill_hex="#2e2d31",
                text_hex="#d8d2cb",
                corner_radius=2,
                font=FontStyle(point_size=8, bold=True),
            ),
            edited=StatusChipStyle(
                fill_hex="#1c3428",
                text_hex="#7fd1ae",
                corner_radius=2,
                font=FontStyle(point_size=8, bold=True),
            ),
        ),
        mute_solo=MuteSoloButtonStyle(
            active=ToggleButtonStateStyle(fill_hex="#28262a", text_hex="#f6f3ee"),
            inactive=ToggleButtonStateStyle(fill_hex="#121018", text_hex="#aaa49e"),
            dimmed_inactive_fill_hex="#121214",
            border_hex="#4a4749",
            corner_radius=2,
            font=FontStyle(point_size=8, bold=True),
        ),
    ),
    take_row=TakeRowStyle(
        row_fill_hex="#171719",
        selected_row_fill_hex="#202022",
        dimmed_row_fill_hex="#121214",
        header_fill_hex="#171719",
        selected_header_fill_hex="#202022",
        dimmed_header_fill_hex="#101010",
        divider_hex="#3a383a",
        label_hex="#aaa49e",
        dimmed_label_hex="#685f67",
        options_button_open_fill_hex="#2e2d31",
        options_button_closed_fill_hex="#202022",
        options_button_dimmed_fill_hex="#18181a",
        options_button_open_text_hex="#e8e2dc",
        options_button_closed_text_hex="#aaa49e",
        options_button_font=FontStyle(point_size=8, bold=True),
        options_area_fill_hex="#101010",
        action_chip=TakeActionChipStyle(
            fill_hex="#202022",
            text_hex="#d8d2cb",
            font=FontStyle(point_size=8),
        ),
    ),
    event_lane=EventLaneStyle(
        default_fill_hex="#885A2D",
        demoted_fill_hex="#685f67",
        dimmed_alpha=120,
        selection_lighten_factor=130,
        border_darkness_factor=160,
        normal_border_width_px=1,
        selected_border_width_px=2,
        text_hex="#101010",
        corner_radius=3,
    ),
    waveform_lane=WaveformLaneStyle(
        dimmed_alpha=120,
        fallback_pen_width_px=1.2,
        cached_pen_width_px=1.0,
        fallback_amp_row_factor=0.30,
        cached_amp_row_factor=0.38,
    ),
    ruler=RulerStyle(
        background_hex="#101010",
        divider_hex="#3a383a",
        split_divider_hex="#5a565a",
        header_background_hex="#101010",
        title_hex="#8f8a84",
        tick_hex="#3d3b3d",
        grid_hex="#242224",
        label_hex="#685f67",
        section_even_hex="#885A2D",
        section_odd_hex="#8f8a84",
        section_alpha=24,
        section_boundary_hex="#d8d2cb",
        section_label_hex="#101010",
        region_even_hex="#28262a",
        region_odd_hex="#242224",
        region_alpha=28,
        region_border_hex="#8f8a84",
        region_label_hex="#c0bab4",
    ),
    fixture=TimelineFixtureStyle(
        default_sync_label="No sync",
        fallback_audio_lane_hex="#8f8a84",
        layer_color_tokens=MappingProxyType(
            {
                "song": "#86a0ad",
                "drums": "#8d7ea5",
                "bass": "#7f9a72",
                "vocals": "#b56f61",
                "other": "#8f8a84",
                "kick": "#a6533e",
                "snare": "#91a19a",
                "hihat": "#b49a63",
                "clap": "#9a8f83",
                "sync": "#7f9a8f",
                "event_preview": "#91a19a",
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


def build_object_palette_stylesheet(
    style: ObjectPaletteStyle = TIMELINE_STYLE.object_palette,
) -> str:
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
            border-radius: 2px;
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
            border-radius: 2px;
            margin: 1px 120px;
        }}
        QPushButton {{
            background: {style.button_bg_hex};
            color: {style.button_fg_hex};
            border: 1px solid {style.button_border_hex};
            border-radius: 2px;
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
            border-radius: 2px;
            padding: 3px 6px;
            min-height: {style.spin_min_height_px}px;
        }}
    """


def build_timeline_scroll_area_stylesheet(style: TimelineShellStyle = TIMELINE_STYLE) -> str:
    return f"background: {style.scroll_area_background_hex}; border: none;"
