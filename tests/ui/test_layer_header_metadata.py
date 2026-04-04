from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock


def test_metadata_tokens_prioritize_core_badges_and_limit_count():
    tokens = LayerHeaderBlock._metadata_tokens(["real-data", "audio", "main", "stem", "event"])
    assert tokens[:4] == ["M", "S", "A", "E"]
    assert "+1" in tokens


def test_metadata_tokens_handle_unknown_badges():
    tokens = LayerHeaderBlock._metadata_tokens(["main", "customthing"])
    assert tokens[0] == "M"
    assert tokens[1] == "C"
