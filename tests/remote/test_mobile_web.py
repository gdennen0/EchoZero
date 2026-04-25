from echozero.remote.mobile_web import build_mobile_web_page


def test_mobile_page_includes_session_and_transport_controls():
    html = build_mobile_web_page()

    assert "Start Session" in html
    assert "Shortcuts" in html
    assert "/api/session/start" in html
    assert "/api/transport/" in html
    assert "/api/audio/current" in html
    assert "Audio Monitor" in html
    assert "shortcut-play" in html
    assert "echozero_remote_token" in html
