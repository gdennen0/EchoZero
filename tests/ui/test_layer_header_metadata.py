from echozero.application.presentation.models import LayerPresentation, LayerStatusPresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId
from echozero.ui.qt.timeline.widget import badge_tooltip_labels
from echozero.ui.qt.timeline.widget import TimelineCanvas


def test_badge_tooltip_labels_expand_readable_text():
    labels = badge_tooltip_labels(["main", "real-data", "classifier-preview"])
    assert labels == ["Main take", "Real data", "Classifier preview"]


def test_badge_tooltip_labels_keep_unknown_readable():
    labels = badge_tooltip_labels(["customthing", "audio"])
    assert labels[0] == "Customthing"
    assert labels[1] == "Audio lane"


def test_header_tooltip_includes_backed_status_and_provenance_metadata():
    layer = LayerPresentation(
        layer_id=LayerId("layer_kick"),
        title="Kick",
        kind=LayerKind.EVENT,
        badges=["main", "event"],
        status=LayerStatusPresentation(
            stale=True,
            manually_modified=True,
            stale_reason="Upstream main changed",
            source_label="stem_separation · drums",
            sync_label="Connected",
            source_layer_id="layer_drums",
            source_song_version_id="version_2",
            pipeline_id="stem_separation",
            output_name="drums",
            source_run_id="run_42",
        ),
    )

    tooltip = TimelineCanvas._header_tooltip_text(layer)

    assert "Main take | Event lane" in tooltip
    assert "Status: Stale (Upstream main changed)" in tooltip
    assert "Status: Manually modified" in tooltip
    assert "stem_separation · drums" in tooltip
    assert "Source layer: layer_drums" in tooltip
    assert "Source song version: version_2" in tooltip
    assert "Pipeline: stem_separation" in tooltip
    assert "Output: drums" in tooltip
    assert "Run: run_42" in tooltip
    assert "Sync: Connected" in tooltip
