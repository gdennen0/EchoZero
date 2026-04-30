from __future__ import annotations

from echozero.foundry.review_web import build_review_page


def test_build_review_page_renders_packaged_assets_without_placeholders() -> None:
    html = build_review_page()

    assert "EZ Review" in html
    assert "Pocket Review Loop" in html
    assert "Open Active Live Review" not in html
    assert 'id="outcome-filter"' not in html
    assert 'id="class-filter"' not in html
    assert 'id="session-drawer"' not in html
    assert 'id="reload-status-btn"' not in html
    assert "Event Layer" in html
    assert "Promote" in html
    assert "Demote" in html
    assert "fetchJson('/api/session?'" in html
    assert "--accent: #43ffa9;" in html
    assert "__REVIEW_PAGE_STYLES__" not in html
    assert "__REVIEW_PAGE_SCRIPT__" not in html
