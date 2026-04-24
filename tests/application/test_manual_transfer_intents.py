from __future__ import annotations

import pytest

from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPreset,
    ApplyTransferPlan,
    CancelTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    DeleteTransferPreset,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    PreviewTransferPlan,
    SaveTransferPreset,
    SelectPullSourceEvents,
    SelectPullSourceTracks,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    SetPullImportMode,
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


def test_select_push_target_track_keeps_optional_layer_id():
    intent = SelectPushTargetTrack(
        target_track_coord="tc1_tg2_tr3",
        layer_id=LayerId("layer_kick"),
    )

    assert intent.target_track_coord == "tc1_tg2_tr3"
    assert intent.layer_id == LayerId("layer_kick")


def test_exit_push_to_ma3_mode_is_constructible():
    intent = ExitPushToMA3Mode()

    assert isinstance(intent, ExitPushToMA3Mode)


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


def test_exit_pull_from_ma3_workspace_is_constructible():
    intent = ExitPullFromMA3Workspace()

    assert isinstance(intent, ExitPullFromMA3Workspace)


def test_select_pull_source_tracks_keeps_source_track_coords():
    intent = SelectPullSourceTracks(source_track_coords=["tc4_tg5_tr6", "tc4_tg5_tr7"])

    assert intent.source_track_coords == ["tc4_tg5_tr6", "tc4_tg5_tr7"]


def test_select_pull_source_tracks_requires_source_track_coords():
    with pytest.raises(
        ValueError,
        match="SelectPullSourceTracks requires at least one source_track_coord",
    ):
        SelectPullSourceTracks(source_track_coords=[])


@pytest.mark.parametrize("source_track_coords", [[""], ["   "], ["tc4_tg5_tr6", ""]])
def test_select_pull_source_tracks_requires_non_empty_source_track_coords(source_track_coords):
    with pytest.raises(
        ValueError,
        match="SelectPullSourceTracks requires non-empty source_track_coords",
    ):
        SelectPullSourceTracks(source_track_coords=source_track_coords)


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


def test_confirm_pull_from_ma3_accepts_main_import_mode():
    intent = ConfirmPullFromMA3(
        source_track_coord="tc4_tg5_tr6",
        selected_ma3_event_ids=["ma3_evt_1"],
        target_layer_id=LayerId("layer_target"),
        import_mode="main",
    )

    assert intent.import_mode == "main"


def test_set_pull_import_mode_accepts_main():
    intent = SetPullImportMode(import_mode="main")

    assert intent.import_mode == "main"


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


def test_transfer_plan_intents_keep_plan_id():
    preview_intent = PreviewTransferPlan(plan_id="plan_123")
    apply_intent = ApplyTransferPlan(plan_id="plan_123")
    cancel_intent = CancelTransferPlan(plan_id="plan_123")

    assert preview_intent.plan_id == "plan_123"
    assert apply_intent.plan_id == "plan_123"
    assert cancel_intent.plan_id == "plan_123"


def test_transfer_preset_intents_keep_validated_fields():
    save_intent = SaveTransferPreset(name=" My Preset ")
    apply_intent = ApplyTransferPreset(preset_id=" preset_123 ")
    delete_intent = DeleteTransferPreset(preset_id=" preset_123 ")

    assert save_intent.name == "My Preset"
    assert apply_intent.preset_id == "preset_123"
    assert delete_intent.preset_id == "preset_123"


@pytest.mark.parametrize(
    ("intent_type", "message"),
    [
        (PreviewTransferPlan, "PreviewTransferPlan requires a non-empty plan_id"),
        (ApplyTransferPlan, "ApplyTransferPlan requires a non-empty plan_id"),
        (CancelTransferPlan, "CancelTransferPlan requires a non-empty plan_id"),
    ],
)
def test_transfer_plan_intents_require_non_empty_plan_id(intent_type, message):
    with pytest.raises(ValueError, match=message):
        intent_type(plan_id="   ")


@pytest.mark.parametrize(
    ("intent_type", "field_name", "message"),
    [
        (SaveTransferPreset, "name", "SaveTransferPreset requires a non-empty name"),
        (ApplyTransferPreset, "preset_id", "ApplyTransferPreset requires a non-empty preset_id"),
        (DeleteTransferPreset, "preset_id", "DeleteTransferPreset requires a non-empty preset_id"),
    ],
)
def test_transfer_preset_intents_require_non_empty_identifiers(intent_type, field_name, message):
    with pytest.raises(ValueError, match=message):
        intent_type(**{field_name: "   "})
