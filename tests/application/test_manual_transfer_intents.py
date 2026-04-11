from __future__ import annotations

import pytest

from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
)


def test_open_push_to_ma3_dialog_captures_selected_event_ids():
    intent = OpenPushToMA3Dialog(
        selection_event_ids=[EventId("evt_1"), EventId("evt_2")],
    )

    assert intent.selection_event_ids == [EventId("evt_1"), EventId("evt_2")]


def test_confirm_push_to_ma3_captures_target_track_and_selected_events():
    intent = ConfirmPushToMA3(
        target_track_coord="tc1_tg2_tr3",
        selected_event_ids=[EventId("evt_1")],
    )

    assert intent.target_track_coord == "tc1_tg2_tr3"
    assert intent.selected_event_ids == [EventId("evt_1")]


@pytest.mark.parametrize("target_track_coord", ["", "   "])
def test_confirm_push_to_ma3_requires_non_empty_target_track_coord(target_track_coord):
    with pytest.raises(
        ValueError,
        match="ConfirmPushToMA3 requires a non-empty target_track_coord",
    ):
        ConfirmPushToMA3(
            target_track_coord=target_track_coord,
            selected_event_ids=[EventId("evt_1")],
        )


def test_confirm_push_to_ma3_requires_selected_event_ids():
    with pytest.raises(
        ValueError,
        match="ConfirmPushToMA3 requires at least one selected_event_id",
    ):
        ConfirmPushToMA3(
            target_track_coord="tc1_tg2_tr3",
            selected_event_ids=[],
        )


def test_open_pull_from_ma3_dialog_is_constructible():
    intent = OpenPullFromMA3Dialog()

    assert isinstance(intent, OpenPullFromMA3Dialog)


def test_select_pull_source_track_keeps_source_track_coord():
    intent = SelectPullSourceTrack(source_track_coord="tc4_tg5_tr6")

    assert intent.source_track_coord == "tc4_tg5_tr6"


@pytest.mark.parametrize("source_track_coord", ["", "   "])
def test_select_pull_source_track_requires_non_empty_source_track_coord(source_track_coord):
    with pytest.raises(
        ValueError,
        match="SelectPullSourceTrack requires a non-empty source_track_coord",
    ):
        SelectPullSourceTrack(source_track_coord=source_track_coord)


def test_select_pull_source_events_keeps_selected_ma3_event_ids():
    intent = SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"])

    assert intent.selected_ma3_event_ids == ["ma3_evt_1", "ma3_evt_2"]


def test_select_pull_source_events_requires_selected_ma3_event_ids():
    with pytest.raises(
        ValueError,
        match="SelectPullSourceEvents requires at least one selected_ma3_event_id",
    ):
        SelectPullSourceEvents(selected_ma3_event_ids=[])


@pytest.mark.parametrize("selected_ma3_event_ids", [[""], ["   "], ["ma3_evt_1", ""]])
def test_select_pull_source_events_requires_non_empty_selected_ma3_event_ids(selected_ma3_event_ids):
    with pytest.raises(
        ValueError,
        match="SelectPullSourceEvents requires non-empty selected_ma3_event_ids",
    ):
        SelectPullSourceEvents(selected_ma3_event_ids=selected_ma3_event_ids)


def test_select_pull_target_layer_keeps_target_layer_id():
    intent = SelectPullTargetLayer(target_layer_id=LayerId("layer_target"))

    assert intent.target_layer_id == LayerId("layer_target")


@pytest.mark.parametrize("target_layer_id", [None, "", "   "])
def test_select_pull_target_layer_requires_non_empty_target_layer_id(target_layer_id):
    with pytest.raises(
        ValueError,
        match="SelectPullTargetLayer requires a non-empty target_layer_id",
    ):
        SelectPullTargetLayer(target_layer_id=target_layer_id)


def test_confirm_pull_from_ma3_defaults_import_mode_and_keeps_fields():
    intent = ConfirmPullFromMA3(
        source_track_coord="tc4_tg5_tr6",
        selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"],
        target_layer_id=LayerId("layer_target"),
    )

    assert intent.source_track_coord == "tc4_tg5_tr6"
    assert intent.selected_ma3_event_ids == ["ma3_evt_1", "ma3_evt_2"]
    assert intent.target_layer_id == LayerId("layer_target")
    assert intent.import_mode == "new_take"


@pytest.mark.parametrize("source_track_coord", ["", "   "])
def test_confirm_pull_from_ma3_requires_non_empty_source_track_coord(source_track_coord):
    with pytest.raises(
        ValueError,
        match="ConfirmPullFromMA3 requires a non-empty source_track_coord",
    ):
        ConfirmPullFromMA3(
            source_track_coord=source_track_coord,
            selected_ma3_event_ids=["ma3_evt_1"],
            target_layer_id=LayerId("layer_target"),
        )


def test_confirm_pull_from_ma3_requires_selected_ma3_event_ids():
    with pytest.raises(
        ValueError,
        match="ConfirmPullFromMA3 requires at least one selected_ma3_event_id",
    ):
        ConfirmPullFromMA3(
            source_track_coord="tc4_tg5_tr6",
            selected_ma3_event_ids=[],
            target_layer_id=LayerId("layer_target"),
        )


@pytest.mark.parametrize("selected_ma3_event_ids", [[""], ["   "], ["ma3_evt_1", ""]])
def test_confirm_pull_from_ma3_requires_non_empty_selected_ma3_event_ids(selected_ma3_event_ids):
    with pytest.raises(
        ValueError,
        match="ConfirmPullFromMA3 requires non-empty selected_ma3_event_ids",
    ):
        ConfirmPullFromMA3(
            source_track_coord="tc4_tg5_tr6",
            selected_ma3_event_ids=selected_ma3_event_ids,
            target_layer_id=LayerId("layer_target"),
        )


@pytest.mark.parametrize("target_layer_id", [None, "", "   "])
def test_confirm_pull_from_ma3_requires_non_empty_target_layer_id(target_layer_id):
    with pytest.raises(
        ValueError,
        match="ConfirmPullFromMA3 requires a non-empty target_layer_id",
    ):
        ConfirmPullFromMA3(
            source_track_coord="tc4_tg5_tr6",
            selected_ma3_event_ids=["ma3_evt_1"],
            target_layer_id=target_layer_id,
        )


def test_apply_pull_from_ma3_is_constructible():
    intent = ApplyPullFromMA3()

    assert isinstance(intent, ApplyPullFromMA3)
