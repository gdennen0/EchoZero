from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock


def test_metadata_text_prioritizes_core_badges_and_limits_width():
    text = LayerHeaderBlock._metadata_text(["real-data", "audio", "main", "stem"])
    assert text.startswith("MAIN")
    assert "STEM" in text
    assert "AUD" in text
    assert "+1" in text


def test_metadata_text_handles_unknown_badges():
    text = LayerHeaderBlock._metadata_text(["main", "customthing"])
    assert "MAIN" in text
    assert "CUST" in text
