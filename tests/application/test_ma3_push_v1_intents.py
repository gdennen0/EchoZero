from __future__ import annotations

import pytest

from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Timecode,
    CreateMA3Track,
    CreateMA3TrackGroup,
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3PushScope,
    MA3PushTargetMode,
    MA3SequenceCreationMode,
    MA3SequenceRefreshRangeMode,
    RefreshMA3PushTracks,
    RefreshMA3Sequences,
    PushLayerToMA3,
    SetLayerMA3Route,
)


def test_push_layer_to_ma3_coerces_string_enums():
    intent = PushLayerToMA3(
        layer_id=LayerId("layer_kick"),
        scope="selected_events",
        target_mode="different_track_once",
        apply_mode="overwrite",
        target_track_coord=" tc1_tg2_tr3 ",
        selected_event_ids=[EventId("evt_1"), EventId("evt_1")],
        sequence_action=CreateMA3Sequence(
            creation_mode="next_available",
            preferred_name=" Kick - Snare ",
        ),
    )

    assert intent.scope is MA3PushScope.SELECTED_EVENTS
    assert intent.target_mode is MA3PushTargetMode.DIFFERENT_TRACK_ONCE
    assert intent.apply_mode is MA3PushApplyMode.OVERWRITE
    assert intent.target_track_coord == "tc1_tg2_tr3"
    assert intent.selected_event_ids == [EventId("evt_1")]
    assert isinstance(intent.sequence_action, CreateMA3Sequence)
    assert intent.sequence_action.creation_mode is MA3SequenceCreationMode.NEXT_AVAILABLE
    assert intent.sequence_action.preferred_name == "Kick - Snare"


def test_refresh_ma3_sequences_coerces_range_mode():
    intent = RefreshMA3Sequences(range_mode="current_song")

    assert intent.range_mode is MA3SequenceRefreshRangeMode.CURRENT_SONG


def test_refresh_ma3_push_tracks_coerces_optional_timecode_and_track_group():
    intent = RefreshMA3PushTracks(
        target_track_coord=" tc2_tg4_tr8 ",
        timecode_no="2",
        track_group_no="4",
    )

    assert intent.target_track_coord == "tc2_tg4_tr8"
    assert intent.timecode_no == 2
    assert intent.track_group_no == 4


def test_create_ma3_timecode_normalizes_optional_name():
    intent = CreateMA3Timecode(preferred_name="  Song A  ")
    assert intent.preferred_name == "Song A"


def test_create_ma3_track_group_requires_positive_timecode():
    with pytest.raises(
        ValueError,
        match="CreateMA3TrackGroup requires integer timecode_no",
    ):
        CreateMA3TrackGroup(timecode_no=None)
    with pytest.raises(
        ValueError,
        match="CreateMA3TrackGroup requires timecode_no >= 1",
    ):
        CreateMA3TrackGroup(timecode_no=0)


def test_create_ma3_track_requires_positive_timecode_and_track_group():
    with pytest.raises(
        ValueError,
        match="CreateMA3Track requires track_group_no >= 1",
    ):
        CreateMA3Track(timecode_no=1, track_group_no=0)
    with pytest.raises(
        ValueError,
        match="CreateMA3Track requires timecode_no >= 1",
    ):
        CreateMA3Track(timecode_no=0, track_group_no=2)


def test_refresh_ma3_push_tracks_requires_timecode_with_track_group():
    with pytest.raises(
        ValueError,
        match="requires timecode_no when track_group_no is provided",
    ):
        RefreshMA3PushTracks(track_group_no=4)


def test_assign_ma3_track_sequence_requires_positive_sequence_no():
    with pytest.raises(
        ValueError,
        match="AssignMA3TrackSequence requires sequence_no >= 1",
    ):
        AssignMA3TrackSequence(
            target_track_coord="tc1_tg2_tr3",
            sequence_no=0,
        )


def test_push_layer_to_ma3_requires_selected_events_for_selected_scope():
    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 requires selected_event_ids for scope 'selected_events'",
    ):
        PushLayerToMA3(
            layer_id=LayerId("layer_kick"),
            scope="selected_events",
            selected_event_ids=[],
        )


def test_push_layer_to_ma3_requires_target_track_for_one_shot_mode():
    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 requires target_track_coord for target_mode 'different_track_once'",
    ):
        PushLayerToMA3(
            layer_id=LayerId("layer_kick"),
            target_mode="different_track_once",
        )


def test_set_layer_ma3_route_requires_non_empty_target_track_coord():
    with pytest.raises(
        ValueError,
        match="SetLayerMA3Route requires a non-empty target_track_coord",
    ):
        SetLayerMA3Route(
            layer_id=LayerId("layer_kick"),
            target_track_coord="   ",
        )
