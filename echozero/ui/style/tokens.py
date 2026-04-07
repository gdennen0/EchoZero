from __future__ import annotations

from dataclasses import dataclass, field

from .scales import ShellScales


@dataclass(frozen=True)
class ShellTokens:
    window_bg: str = "#12151b"
    canvas_bg: str = "#12151b"
    panel_bg: str = "#0d1117"
    panel_alt_bg: str = "#141a23"
    panel_border: str = "#202632"
    section_border: str = "#242d39"
    control_bg: str = "#1b2230"
    control_bg_disabled: str = "#11161d"
    control_bg_active: str = "#264868"
    control_border: str = "#2d3747"
    control_border_active: str = "#4f81b8"
    control_text: str = "#eef3fb"
    control_text_disabled: str = "#657286"
    text_primary: str = "#f3f6fb"
    text_secondary: str = "#9aa6ba"
    slider_handle: str = "#d7e3f4"
    scales: ShellScales = field(default_factory=ShellScales)


SHELL_TOKENS = ShellTokens()
