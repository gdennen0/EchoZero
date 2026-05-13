"""Visual Lab nested style target maps.
Exists so widgets can expose component -> part -> property style addresses.
Representative mappings use current tokens and can expand toward exhaustive coverage.
"""

from __future__ import annotations

from dev.visual_lab.catalog import StylePropertySpec, StyleTargetSpec


def style_target(
    component: str,
    part_path: str,
    label: str,
    properties: tuple[tuple[str, str], ...],
) -> StyleTargetSpec:
    """Build a nested style target from property names and token paths."""
    return StyleTargetSpec(
        component=component,
        part_path=part_path,
        label=label,
        properties=tuple(StylePropertySpec(name, token_path) for name, token_path in properties),
    )


GLOBAL_COLOR_TARGETS = (
    style_target(
        "theme.global_colors",
        "surface",
        "Global surfaces",
        (
            ("app_background", "global_colors.app_background"),
            ("surface", "global_colors.surface"),
            ("surface_raised", "global_colors.surface_raised"),
        ),
    ),
    style_target(
        "theme.global_colors",
        "text",
        "Global text",
        (
            ("primary", "global_colors.text_primary"),
            ("secondary", "global_colors.text_secondary"),
        ),
    ),
    style_target(
        "theme.global_colors",
        "semantic",
        "Global semantic colors",
        (
            ("primary", "global_colors.primary"),
            ("secondary", "global_colors.secondary"),
            ("success", "global_colors.success"),
            ("warning", "global_colors.warning"),
            ("error", "global_colors.error"),
        ),
    ),
)


TIMELINE_LAYER_ROW_TARGETS = (
    style_target(
        "timeline.layer_row",
        "background",
        "Row background",
        (
            ("fill", "palette.row"),
            ("child_fill", "palette.row_child"),
            ("selected_fill", "palette.row_selected"),
            ("muted_fill", "palette.row_muted"),
            ("separator", "palette.border"),
            ("height", "metrics.audio_row_height_px"),
            ("child_height", "metrics.stem_row_height_px"),
        ),
    ),
    style_target(
        "timeline.layer_row",
        "header.title",
        "Header title",
        (
            ("color", "palette.text"),
            ("muted_color", "palette.text_muted"),
            ("font_family", "fonts.family"),
            ("font_size", "fonts.label_px"),
            ("header_width", "metrics.timeline_header_width_px"),
        ),
    ),
    style_target(
        "timeline.layer_row",
        "header.badge",
        "Header badges",
        (
            ("ok_fill", "palette.status_ok"),
            ("sync_fill", "palette.status_sync"),
            ("stale_fill", "palette.status_stale"),
            ("muted_fill", "palette.status_muted"),
            ("height", "metrics.status_chip_height_px"),
        ),
    ),
    style_target(
        "timeline.layer_row",
        "canvas.waveform",
        "Canvas waveform",
        (
            ("line", "palette.waveform"),
            ("fill", "palette.waveform_fill"),
            ("grid", "palette.grid"),
            ("accent", "palette.accent"),
            ("height", "metrics.waveform_height_px"),
        ),
    ),
)

TIMELINE_HEADER_TARGETS = (
    style_target(
        "timeline.layer_header",
        "surface",
        "Header surface",
        (
            ("fill", "palette.panel"),
            ("raised_fill", "palette.panel_raised"),
            ("border", "palette.border"),
            ("width", "metrics.timeline_header_width_px"),
        ),
    ),
    style_target(
        "timeline.layer_header",
        "title",
        "Header title",
        (
            ("color", "palette.text"),
            ("muted_color", "palette.text_muted"),
            ("font_family", "fonts.family"),
            ("font_size", "fonts.label_px"),
        ),
    ),
    style_target(
        "timeline.layer_header",
        "badge.border",
        "Badge border",
        (
            ("border", "palette.border"),
            ("radius", "metrics.control_radius_px"),
            ("height", "metrics.status_chip_height_px"),
        ),
    ),
)

TRANSPORT_TARGETS = (
    style_target(
        "transport",
        "surface",
        "Transport surface",
        (
            ("fill", "palette.panel_raised"),
            ("border", "palette.border"),
            ("radius", "metrics.corner_radius_px"),
        ),
    ),
    style_target(
        "transport.button",
        "play.icon",
        "Play button icon",
        (
            ("color", "palette.text"),
            ("active_color", "palette.accent"),
            ("font_family", "fonts.family"),
            ("font_size", "fonts.label_px"),
        ),
    ),
    style_target(
        "transport.status",
        "stale.badge",
        "Stale status badge",
        (
            ("fill", "palette.status_stale"),
            ("text", "palette.text"),
            ("height", "metrics.status_chip_height_px"),
        ),
    ),
)

STATUS_CHIP_TARGETS = (
    style_target(
        "status_chip",
        "badge.fill",
        "Badge fills",
        (
            ("ok", "palette.status_ok"),
            ("sync", "palette.status_sync"),
            ("stale", "palette.status_stale"),
            ("muted", "palette.status_muted"),
        ),
    ),
    style_target(
        "status_chip",
        "badge.border",
        "Badge border",
        (
            ("border", "palette.border"),
            ("height", "metrics.status_chip_height_px"),
        ),
    ),
    style_target(
        "status_chip",
        "label",
        "Badge label",
        (
            ("color", "palette.text"),
            ("font_family", "fonts.family"),
            ("font_size", "fonts.small_px"),
        ),
    ),
)

CONTROL_TARGETS = (
    style_target(
        "control_card",
        "surface",
        "Card surface",
        (
            ("fill", "palette.panel_raised"),
            ("border", "palette.border"),
            ("radius", "metrics.corner_radius_px"),
            ("padding", "metrics.padding_px"),
        ),
    ),
    style_target(
        "control_card.button",
        "primary",
        "Primary button",
        (
            ("fill", "palette.accent"),
            ("text", "palette.text"),
            ("radius", "metrics.control_radius_px"),
        ),
    ),
    style_target(
        "control_card.button",
        "danger",
        "Danger button",
        (
            ("fill", "palette.danger"),
            ("text", "palette.text"),
            ("radius", "metrics.control_radius_px"),
        ),
    ),
)

WAVEFORM_TARGETS = (
    style_target(
        "waveform",
        "plot.line",
        "Waveform line",
        (("color", "palette.waveform"), ("height", "metrics.waveform_height_px")),
    ),
    style_target(
        "waveform",
        "plot.fill",
        "Waveform fill",
        (("color", "palette.waveform_fill"), ("surface", "palette.panel")),
    ),
)
