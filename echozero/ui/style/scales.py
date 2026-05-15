from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellScales:
    panel_padding: int = 8
    section_padding: int = 5
    section_gap: int = 4
    layout_gap: int = 5
    inline_gap: int = 4
    compact_gap: int = 2
    field_padding_v: int = 3
    field_padding_h: int = 6
    button_radius: int = 3
    panel_radius: int = 3
    slider_groove_height: int = 4
    slider_handle_width: int = 12
    slider_handle_margin: int = -5
    border_width: int = 1
