"""Visual Lab theme catalog entries.
Exists so global theme objects stay separate from production-backed entry factories.
Used by the support-only Visual Lab catalog registry.
"""

from __future__ import annotations

from dev.visual_lab.catalog import CatalogEntry
from dev.visual_lab.catalog_sources import LAB_PRIMITIVE_SOURCE
from dev.visual_lab.editable_tokens import GLOBAL_COLOR_TOKENS
from dev.visual_lab.style_targets import GLOBAL_COLOR_TARGETS
from dev.visual_lab.style_widgets import GlobalColorPalettePreviewWidget
from dev.visual_lab.widgets import CatalogFrame


def theme_entries() -> tuple[CatalogEntry, ...]:
    """Build Visual Lab catalog entries for global theme editing."""
    return (
        CatalogEntry(
            entry_id="theme.global-colors",
            name="Global colors",
            category="Theme",
            description="Shared global color decisions used as the base for lab previews.",
            kind="primitive",
            source_kind="lab-only experimental",
            source_path=LAB_PRIMITIVE_SOURCE,
            render=lambda tokens: CatalogFrame(
                tokens,
                "Global colors",
                GlobalColorPalettePreviewWidget(tokens),
                width=720,
                height=260,
            ),
            editable_token_paths=GLOBAL_COLOR_TOKENS,
            style_targets=GLOBAL_COLOR_TARGETS,
        ),
    )
