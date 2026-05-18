"""Timeline command runtime classification tests.
Exists to keep app-shell edit metadata scoped around hot timeline mutations.
Connects section edits to bounded history and storage behavior.
"""

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, SongVersionId, TimelineId
from echozero.application.timeline.intents import ReplaceSectionCues, SectionCueEdit
from echozero.application.timeline.models import Layer, Timeline
from echozero.ui.qt.timeline_command_runtime import TimelineCommandRuntime


def test_replace_section_cues_scopes_history_and_storage_to_target_section_layer() -> None:
    timeline_id = TimelineId("timeline_sections")
    section_layer_id = LayerId("layer_sections")
    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId("song_version"),
        layers=[
            Layer(
                id=LayerId("layer_audio"),
                timeline_id=timeline_id,
                name="Audio",
                kind=LayerKind.AUDIO,
                order_index=0,
            ),
            Layer(
                id=section_layer_id,
                timeline_id=timeline_id,
                name="Sections",
                kind=LayerKind.SECTION,
                order_index=1,
            ),
        ],
    )

    result = TimelineCommandRuntime().prepare(
        timeline,
        ReplaceSectionCues(
            target_layer_id=section_layer_id,
            cues=[SectionCueEdit(cue_id=None, start=12.0, name="Verse")],
        ),
    )

    assert result.scoped_history is True
    assert result.history_layer_ids == (section_layer_id,)
    assert result.storage_layer_ids == (section_layer_id,)

