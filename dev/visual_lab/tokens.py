"""Visual Lab tokens: human-editable theme vocabulary for preview surfaces.
Exists so aggressive visual iteration can happen outside canonical runtime styling.
Loaded by the lab preview, capture tooling, and focused token tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
import re
import tomllib
from typing import Literal

DEFAULT_TOKEN_PATH = Path(__file__).resolve().with_name("tokens.toml")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
TokenSection = Literal["global_colors", "palette", "fonts", "metrics"]


@dataclass(frozen=True, slots=True)
class VisualLabGlobalColors:
    """Global color tokens intended for shared Visual Lab theme decisions."""

    app_background: str
    surface: str
    surface_raised: str
    text_primary: str
    text_secondary: str
    primary: str
    secondary: str
    success: str
    warning: str
    error: str


@dataclass(frozen=True, slots=True)
class VisualLabPalette:
    """Color tokens used by Visual Lab preview scenes."""

    window: str
    panel: str
    panel_raised: str
    row: str
    row_child: str
    row_selected: str
    row_muted: str
    border: str
    grid: str
    text: str
    text_muted: str
    accent: str
    accent_secondary: str
    waveform: str
    waveform_fill: str
    status_ok: str
    status_sync: str
    status_stale: str
    status_muted: str
    danger: str


@dataclass(frozen=True, slots=True)
class VisualLabFonts:
    """Font tokens used by Visual Lab preview scenes."""

    family: str
    mono_family: str
    base_px: int
    small_px: int
    label_px: int
    title_px: int
    weight_regular: int
    weight_medium: int
    weight_bold: int


@dataclass(frozen=True, slots=True)
class VisualLabMetrics:
    """Size tokens used by Visual Lab preview scenes."""

    corner_radius_px: int
    control_radius_px: int
    border_width_px: int
    glow_strength_px: int
    timeline_header_width_px: int
    timeline_width_px: int
    audio_row_height_px: int
    stem_row_height_px: int
    transport_height_px: int
    status_chip_height_px: int
    gap_px: int
    padding_px: int
    waveform_height_px: int


@dataclass(frozen=True, slots=True)
class VisualLabTokens:
    """Complete Visual Lab theme vocabulary."""

    global_colors: VisualLabGlobalColors
    palette: VisualLabPalette
    fonts: VisualLabFonts
    metrics: VisualLabMetrics


@dataclass(frozen=True, slots=True)
class TokenFieldSpec:
    """One editable Visual Lab token field."""

    section: TokenSection
    name: str
    path: str
    value_type: type
    value: str | int


def load_tokens(path: str | Path | None = None) -> VisualLabTokens:
    """Load Visual Lab tokens from a TOML file."""
    token_path = Path(path) if path is not None else DEFAULT_TOKEN_PATH
    payload = tomllib.loads(token_path.read_text(encoding="utf-8"))
    palette_payload = payload["palette"]
    global_colors_payload = payload.get("global_colors")
    if global_colors_payload is None:
        global_colors_payload = _global_colors_from_palette_payload(palette_payload)
    tokens = VisualLabTokens(
        global_colors=VisualLabGlobalColors(**global_colors_payload),
        palette=VisualLabPalette(**payload["palette"]),
        fonts=VisualLabFonts(**payload["fonts"]),
        metrics=VisualLabMetrics(**payload["metrics"]),
    )
    _validate_tokens(tokens)
    return tokens


def save_tokens(tokens: VisualLabTokens, path: str | Path | None = None) -> None:
    """Save Visual Lab tokens to a simple TOML file."""
    _validate_tokens(tokens)
    token_path = Path(path) if path is not None else DEFAULT_TOKEN_PATH
    token_path.write_text(_format_tokens_toml(tokens), encoding="utf-8")


def token_field_specs(tokens: VisualLabTokens) -> tuple[TokenFieldSpec, ...]:
    """Return editable token field metadata for generated lab forms."""
    specs: list[TokenFieldSpec] = []
    for section in ("global_colors", "palette", "fonts", "metrics"):
        section_value = getattr(tokens, section)
        for field in fields(section_value):
            value = getattr(section_value, field.name)
            specs.append(
                TokenFieldSpec(
                    section=section,
                    name=field.name,
                    path=f"{section}.{field.name}",
                    value_type=type(value),
                    value=value,
                )
            )
    return tuple(specs)


def update_token_values(
    tokens: VisualLabTokens,
    values: Iterable[tuple[str, str | int]],
) -> VisualLabTokens:
    """Return tokens with edited path values applied and validated."""
    section_values = {
        "global_colors": {
            field.name: getattr(tokens.global_colors, field.name)
            for field in fields(tokens.global_colors)
        },
        "palette": {
            field.name: getattr(tokens.palette, field.name) for field in fields(tokens.palette)
        },
        "fonts": {field.name: getattr(tokens.fonts, field.name) for field in fields(tokens.fonts)},
        "metrics": {
            field.name: getattr(tokens.metrics, field.name) for field in fields(tokens.metrics)
        },
    }
    field_types = {spec.path: spec.value_type for spec in token_field_specs(tokens)}
    for path, raw_value in values:
        section, field_name = _split_token_path(path)
        value_type = field_types[path]
        section_values[section][field_name] = _coerce_token_value(path, raw_value, value_type)

    updated = VisualLabTokens(
        global_colors=VisualLabGlobalColors(**section_values["global_colors"]),
        palette=VisualLabPalette(**section_values["palette"]),
        fonts=VisualLabFonts(**section_values["fonts"]),
        metrics=VisualLabMetrics(**section_values["metrics"]),
    )
    _validate_tokens(updated)
    return updated


def _format_tokens_toml(tokens: VisualLabTokens) -> str:
    lines = [
        "# EchoZero Visual Lab tokens.",
        "# Edit here or use: python -m dev.visual_lab.preview",
        "",
    ]
    for section in ("global_colors", "palette", "fonts", "metrics"):
        section_value = getattr(tokens, section)
        lines.append(f"[{section}]")
        for field in fields(section_value):
            value = getattr(section_value, field.name)
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{field.name} = "{escaped}"')
            else:
                lines.append(f"{field.name} = {value}")
        lines.append("")
    return "\n".join(lines)


def _split_token_path(path: str) -> tuple[TokenSection, str]:
    parts = str(path).split(".", 1)
    if len(parts) != 2 or parts[0] not in {"global_colors", "palette", "fonts", "metrics"}:
        raise ValueError(f"unknown Visual Lab token path: {path}")
    section = parts[0]
    section_fields = {
        "global_colors": VisualLabGlobalColors,
        "palette": VisualLabPalette,
        "fonts": VisualLabFonts,
        "metrics": VisualLabMetrics,
    }
    field_names = {field.name for field in fields(section_fields[section])}
    if parts[1] not in field_names:
        raise ValueError(f"unknown Visual Lab token path: {path}")
    return section, parts[1]


def _coerce_token_value(path: str, raw_value: str | int, value_type: type) -> str | int:
    if value_type is int:
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be an integer") from exc
    return str(raw_value)


def _validate_tokens(tokens: VisualLabTokens) -> None:
    global_colors = tokens.global_colors
    for field_name in VisualLabGlobalColors.__dataclass_fields__:
        value = getattr(global_colors, field_name)
        if not _HEX_COLOR_RE.match(value):
            raise ValueError(f"global_colors.{field_name} must be a #RRGGBB color")

    palette = tokens.palette
    for field_name in VisualLabPalette.__dataclass_fields__:
        value = getattr(palette, field_name)
        if not _HEX_COLOR_RE.match(value):
            raise ValueError(f"palette.{field_name} must be a #RRGGBB color")

    metrics = tokens.metrics
    for field_name in VisualLabMetrics.__dataclass_fields__:
        value = getattr(metrics, field_name)
        if value < 0:
            raise ValueError(f"metrics.{field_name} must be non-negative")


def _global_colors_from_palette_payload(payload: dict[str, str]) -> dict[str, str]:
    return {
        "app_background": payload["window"],
        "surface": payload["panel"],
        "surface_raised": payload["panel_raised"],
        "text_primary": payload["text"],
        "text_secondary": payload["text_muted"],
        "primary": payload["accent"],
        "secondary": payload["accent_secondary"],
        "success": payload["status_ok"],
        "warning": payload["status_stale"],
        "error": payload["danger"],
    }
