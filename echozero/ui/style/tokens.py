from __future__ import annotations

from dataclasses import dataclass, field

from .scales import ShellScales


@dataclass(frozen=True)
class ShellTokens:
    window_bg: str = "#101010"
    canvas_bg: str = "#101010"
    panel_bg: str = "#121214"
    panel_alt_bg: str = "#171719"
    panel_border: str = "#2e2d31"
    section_border: str = "#3a383a"
    control_bg: str = "#202022"
    control_bg_disabled: str = "#18181a"
    control_bg_active: str = "#28262a"
    control_border: str = "#4a4749"
    control_border_active: str = "#8f8a84"
    control_text: str = "#e8e2dc"
    control_text_disabled: str = "#807a74"
    text_primary: str = "#f6f3ee"
    text_secondary: str = "#aaa49e"
    success_bg: str = "#1c3428"
    success_border: str = "#3d7d5b"
    danger_bg: str = "#301913"
    danger_border: str = "#8f3a2f"
    slider_handle: str = "#d8d2cb"
    scales: ShellScales = field(default_factory=ShellScales)


SHELL_TOKENS = ShellTokens()
