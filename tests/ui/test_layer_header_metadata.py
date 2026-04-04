from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock
from echozero.ui.qt.timeline.widget import badge_tooltip_labels


def test_metadata_tokens_prioritize_core_badges_and_limit_count():
    tokens = LayerHeaderBlock._metadata_tokens(["real-data", "audio", "main", "stem", "event"])
    assert tokens[:4] == ["M", "S", "A", "E"]
    assert "+1" in tokens


def test_metadata_tokens_handle_unknown_badges():
    tokens = LayerHeaderBlock._metadata_tokens(["main", "customthing"])
    assert tokens[0] == "M"
    assert tokens[1] == "C"


def test_badge_tooltip_labels_expand_readable_text():
    labels = badge_tooltip_labels(["main", "real-data", "classifier-preview"])
    assert labels == ["Main take", "Real data", "Classifier preview"]
