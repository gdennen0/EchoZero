from echozero.ui.qt.timeline.widget import badge_tooltip_labels


def test_badge_tooltip_labels_expand_readable_text():
    labels = badge_tooltip_labels(["main", "real-data", "classifier-preview"])
    assert labels == ["Main take", "Real data", "Classifier preview"]


def test_badge_tooltip_labels_keep_unknown_readable():
    labels = badge_tooltip_labels(["customthing", "audio"])
    assert labels[0] == "Customthing"
    assert labels[1] == "Audio lane"
