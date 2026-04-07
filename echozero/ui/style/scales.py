from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellScales:
    panel_padding: int = 14
    section_padding: int = 10
    section_gap: int = 8
    layout_gap: int = 10
    inline_gap: int = 6
    compact_gap: int = 4
    field_padding_v: int = 6
    field_padding_h: int = 10
    button_radius: int = 7
    panel_radius: int = 10
    slider_groove_height: int = 4
    slider_handle_width: int = 12
    slider_handle_margin: int = -5
    border_width: int = 1
