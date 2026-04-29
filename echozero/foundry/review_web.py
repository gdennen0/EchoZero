"""EZ Review web page assembly helpers.
Exists because operators need a clear current-project review lane without a monolithic inline blob.
Connects packaged HTML/CSS/JS assets to the review-session HTTP surface.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

_ASSET_PACKAGE = "echozero.foundry.web_assets"
_STYLE_PLACEHOLDER = "__REVIEW_PAGE_STYLES__"
_SCRIPT_PLACEHOLDER = "__REVIEW_PAGE_SCRIPT__"


def build_review_page() -> str:
    """Return the single-screen EZ Review page."""

    return _render_review_page(
        template=_read_asset_text("review_page.html"),
        styles=_read_asset_text("review_page.css"),
        script=_read_asset_text("review_page.js"),
    )


def _render_review_page(*, template: str, styles: str, script: str) -> str:
    if _STYLE_PLACEHOLDER not in template:
        raise ValueError("review page template is missing the style placeholder")
    if _SCRIPT_PLACEHOLDER not in template:
        raise ValueError("review page template is missing the script placeholder")
    return template.replace(_STYLE_PLACEHOLDER, styles, 1).replace(_SCRIPT_PLACEHOLDER, script, 1)


@lru_cache(maxsize=None)
def _read_asset_text(filename: str) -> str:
    return files(_ASSET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
