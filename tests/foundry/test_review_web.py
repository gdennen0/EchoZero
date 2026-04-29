from __future__ import annotations

from echozero.foundry.review_web import build_review_page


def test_build_review_page_renders_packaged_assets_without_placeholders() -> None:
    html = build_review_page()

    assert "EZ Review" in html
    assert "Pocket Review Loop" in html
    assert "const initialQuery = new URLSearchParams(window.location.search);" in html
    assert "--accent: #43ffa9;" in html
    assert "__REVIEW_PAGE_STYLES__" not in html
    assert "__REVIEW_PAGE_SCRIPT__" not in html
