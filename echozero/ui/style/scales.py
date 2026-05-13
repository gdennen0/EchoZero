from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellScales:
    panel_padding: int = 12
    section_padding: int = 8
    section_gap: int = 6
    layout_gap: int = 8
    inline_gap: int = 6
    compact_gap: int = 4
    field_padding_v: int = 6
    field_padding_h: int = 10
    button_radius: int = 6
    panel_radius: int = 8
    slider_groove_height: int = 4
    slider_handle_width: int = 12
    slider_handle_margin: int = -5
    border_width: int = 1
