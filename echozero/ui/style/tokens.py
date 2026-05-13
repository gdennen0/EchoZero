from __future__ import annotations

from dataclasses import dataclass, field

from .scales import ShellScales


@dataclass(frozen=True)
class ShellTokens:
    window_bg: str = "#12151b"
    canvas_bg: str = "#12151b"
    panel_bg: str = "#0d1117"
    panel_alt_bg: str = "#111720"
    panel_border: str = "#1c2430"
    section_border: str = "#222b38"
    control_bg: str = "#18212c"
    control_bg_disabled: str = "#11161d"
    control_bg_active: str = "#274662"
    control_border: str = "#2a3443"
    control_border_active: str = "#5a84b2"
    control_text: str = "#eef3fb"
    control_text_disabled: str = "#657286"
    text_primary: str = "#f3f6fb"
    text_secondary: str = "#9aa6ba"
    success_bg: str = "#1c3428"
    success_border: str = "#3d7d5b"
    danger_bg: str = "#341d24"
    danger_border: str = "#8b4451"
    slider_handle: str = "#d7e3f4"
    scales: ShellScales = field(default_factory=ShellScales)


SHELL_TOKENS = ShellTokens()
