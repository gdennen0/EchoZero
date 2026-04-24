from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
    render_inspector_contract_text,
)
from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    ManualPullEventOptionPresentation,
    RegionPresentation,
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, RegionId, TakeId, TimelineId
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.object_actions import descriptor_for_action, is_object_action


def _contract_test_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_contract"),
        title="Contract",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_kick"),
                title="Kick",
                main_take_id=TakeId("take_main"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("main_evt"),
                        start=1.0,
                        end=1.5,
                        label="Main",
                    )
                ],
                takes=[
                    TakeLanePresentation(
                        take_id=TakeId("take_alt"),
                        name="Take 2",
                        kind=LayerKind.EVENT,
                        events=[
                            EventPresentation(
                                event_id=EventId("take_evt"),
                                start=2.0,
                                end=2.5,
                                label="Take",
                            )
                        ],
                    )
                ],
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_empty"),
                title="Empty",
                main_take_id=None,
                kind=LayerKind.EVENT,
                events=[],
                takes=[],
                status=LayerStatusPresentation(),
            ),
        ],
        end_time_label="00:05.00",
    )


def _song_version_contract_presentation() -> TimelinePresentation:
    presentation = _contract_test_presentation()
    presentation.active_song_id = "song_alpha"
    presentation.active_song_title = "Alpha Song"
    presentation.active_song_version_id = "song_version_festival"
    presentation.active_song_version_label = "Festival Edit"
    presentation.available_songs = [
        SongOptionPresentation(
            song_id="song_alpha",
            title="Alpha Song",
            is_active=True,
            active_version_id="song_version_festival",
            active_version_label="Festival Edit",
            version_count=2,
            versions=[
                SongVersionOptionPresentation(
                    song_version_id="song_version_original",
                    label="Original",
                ),
                SongVersionOptionPresentation(
                    song_version_id="song_version_festival",
                    label="Festival Edit",
                    is_active=True,
                ),
            ],
        ),
        SongOptionPresentation(
            song_id="song_beta",
            title="Beta Song",
            active_version_id="song_version_beta",
            active_version_label="Original",
            version_count=1,
            versions=[
                SongVersionOptionPresentation(
                    song_version_id="song_version_beta",
                    label="Original",
                    is_active=True,
                )
            ],
        ),
    ]
    presentation.available_song_versions = [
        SongVersionOptionPresentation(song_version_id="song_version_original", label="Original"),
        SongVersionOptionPresentation(
            song_version_id="song_version_festival",
            label="Festival Edit",
            is_active=True,
        ),
    ]
    return presentation


def _section_rows(contract):
    return {row.label: row.value for section in contract.sections for row in section.rows}


def test_inspector_contract_no_selection_state():
    contract = build_timeline_inspector_contract(_contract_test_presentation())
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]

    assert contract.identity is None
    assert contract.title == "No timeline object selected."
    assert render_inspector_contract_text(contract) == "No timeline object selected."
    assert "send_layer_to_ma3" not in action_ids
    assert "add_event_layer" in action_ids
    assert "add_automation_layer" not in action_ids
    assert "add_reference_layer" not in action_ids


def test_inspector_contract_falls_back_to_current_song_version_without_selection():
    contract = build_timeline_inspector_contract(_song_version_contract_presentation())
    rows = _section_rows(contract)
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }

    assert contract.identity is not None
    assert contract.identity.object_type == "song_version"
    assert contract.title == "Song Alpha Song"
    assert rows["song id"] == "song_alpha"
    assert rows["song title"] == "Alpha Song"
    assert rows["version id"] == "song_version_festival"
    assert rows["version label"] == "Festival Edit"
    assert rows["ma3 tc pool"] == "unconfigured"
    assert rows["versions"] == "2"
    assert rows["timeline duration"] == "00:05.00"
    assert rows["layers"] == "2"
    assert {
        "song.select",
        "song.version.switch",
        "song.version.add",
        "song.delete",
        "song.version.delete",
        "song.add",
    } <= action_ids


def test_inspector_contract_layer_selection_state():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.layers[0].sync_target_label = "tc1_tg2_tr3"

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]
    section_ids = [section.section_id for section in contract.context_sections]

    assert contract.identity is not None
    assert contract.identity.object_type == "layer"
    assert contract.title == "Layer Kick"
    assert rows["id"] == "layer_kick"
    assert rows["kind"] == "EVENT"
    assert rows["main take"] == "take_main"
    assert rows["status flags"] == "none"
    assert rows["playback state"] == "Set Active"
    assert rows["sync state"] == "Off"
    assert rows["sync mapping"] == "tc1_tg2_tr3"
    assert rows["selected identity"] == "Layer Kick (layer_kick)"
    assert rows["playback target"] == "none"
    assert {"set_active_playback_target", "gain_down", "gain_unity", "gain_up"} <= set(action_ids)
    assert {
        "selection.select_every_other",
        "selection.renumber_cues_from_one",
    } <= set(action_ids)
    assert {
        "route_layer_to_ma3_track",
        "send_layer_to_ma3",
        "send_selected_events_to_ma3",
        "send_to_different_track_once",
    } <= set(action_ids)
    assert "event-batch" in section_ids
    assert "sync-transfer" in section_ids
    assert "live-sync" not in [section.section_id for section in contract.context_sections]


def test_inspector_contract_layer_selection_keeps_song_switching_actions_available():
    presentation = _song_version_contract_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")

    contract = build_timeline_inspector_contract(presentation)
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }

    assert {"song.select", "song.version.switch", "song.version.add"} <= action_ids
    assert "song.add" not in action_ids


def test_inspector_contract_object_actions_are_registered_descriptors():
    presentation = _contract_test_presentation()
    presentation.layers[0].kind = LayerKind.AUDIO
    presentation.selected_layer_id = LayerId("layer_kick")

    contract = build_timeline_inspector_contract(presentation)
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]
    object_action_ids = [
        action_id for action_id in action_ids if action_id.startswith("timeline.")
    ]

    assert object_action_ids
    for action_id in object_action_ids:
        descriptor = descriptor_for_action(action_id)
        assert descriptor is not None
        assert is_object_action(action_id) is True


def test_inspector_contract_audio_layer_hides_ma3_controls():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.layers[0].kind = LayerKind.AUDIO

    contract = build_timeline_inspector_contract(presentation)
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }
    section_ids = [section.section_id for section in contract.context_sections]

    assert "sync-transfer" not in section_ids
    assert "route_layer_to_ma3_track" not in action_ids
    assert "send_layer_to_ma3" not in action_ids
    assert "send_selected_events_to_ma3" not in action_ids
    assert "send_to_different_track_once" not in action_ids


def test_inspector_contract_hides_legacy_transfer_surface_even_if_legacy_state_exists():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.manual_push_flow.push_mode_active = True
    presentation.manual_pull_flow.workspace_active = True
    presentation.batch_transfer_plan = BatchTransferPlanPresentation(
        plan_id="push:timeline_contract",
        operation_type="push",
        rows=[
            BatchTransferPlanRowPresentation(
                row_id="push:layer_kick",
                direction="push",
                source_label="Kick",
                target_label="Track 3",
                selected_count=1,
                status="ready",
            )
        ],
        ready_count=1,
    )

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }

    assert "push mode" not in rows
    assert "pull workspace" not in rows
    assert "transfer plan" not in rows
    assert "transfer.plan_preview" not in action_ids
    assert "transfer.plan_apply" not in action_ids
    assert "transfer.plan_cancel" not in action_ids
    assert "pull_from_ma3" in action_ids


def test_inspector_contract_hides_transfer_preset_actions_from_primary_transfer_surface():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.manual_push_flow.push_mode_active = True
    presentation.layers[0].sync_target_label = "tc1_tg2_tr3"

    contract = build_timeline_inspector_contract(presentation)
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }

    assert "save_transfer_preset" not in action_ids
    assert "apply_transfer_preset" not in action_ids
    assert "delete_transfer_preset" not in action_ids


def test_inspector_contract_live_sync_section_hidden_when_experimental_disabled():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.layers[0].live_sync_state = LiveSyncState.OBSERVE
    presentation.layers[0].live_sync_pause_reason = "operator pause"
    presentation.layers[0].live_sync_divergent = True

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    section_ids = [section.section_id for section in contract.context_sections]

    assert "live-sync" not in section_ids
    assert "live sync state" not in rows
    assert "live sync pause" not in rows
    assert "live sync divergence" not in rows


def test_inspector_contract_live_sync_section_visible_when_experimental_enabled():
    presentation = _contract_test_presentation()
    presentation.experimental_live_sync_enabled = True
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.layers[0].live_sync_state = LiveSyncState.PAUSED
    presentation.layers[0].live_sync_pause_reason = "operator pause"
    presentation.layers[0].live_sync_divergent = True

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    live_sync_section = next(
        section for section in contract.context_sections if section.section_id == "live-sync"
    )
    action_ids = [action.action_id for action in live_sync_section.actions]

    assert rows["live sync state"] == "paused"
    assert rows["live sync pause"] == "operator pause"
    assert rows["live sync divergence"] == "diverged"
    assert action_ids == [
        "live_sync_set_off",
        "live_sync_set_observe",
        "live_sync_set_armed_write",
        "live_sync_set_pause_reason",
        "live_sync_clear_pause_reason",
    ]


def test_inspector_contract_main_event_state():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.selected_take_id = TakeId("take_main")
    presentation.selected_event_ids = [EventId("main_evt")]
    presentation.layers[0].sync_target_label = "tc1_tg2_tr3"

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]
    transfer_section = next(
        section for section in contract.sections if section.section_id == "event-transfer"
    )

    assert contract.identity is not None
    assert contract.identity.object_type == "event"
    assert contract.title == "Event Main"
    assert rows["id"] == "main_evt"
    assert rows["start"] == "1.00s"
    assert rows["end"] == "1.50s"
    assert rows["duration"] == "0.50s"
    assert rows["take"] == "Main take (take_main)"
    assert rows["playback state"] == "Set Active"
    assert rows["sync mapping"] == "tc1_tg2_tr3"
    assert rows["selected identity"] == "Event Main (main_evt) on Kick / Main take (take_main)"
    assert rows["playback target"] == "none"
    assert transfer_section.label == "Sync & Transfer"
    assert {
        "selection.select_every_other",
        "selection.renumber_cues_from_one",
    } <= set(action_ids)
    assert {
        "route_layer_to_ma3_track",
        "send_layer_to_ma3",
        "send_selected_events_to_ma3",
        "send_to_different_track_once",
    } <= set(action_ids)


def test_inspector_contract_take_event_state():
    presentation = _contract_test_presentation()
    presentation.active_playback_layer_id = LayerId("layer_kick")
    presentation.active_playback_take_id = TakeId("take_alt")

    contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(
            kind="event",
            layer_id=LayerId("layer_kick"),
            take_id=TakeId("take_alt"),
            event_id=EventId("take_evt"),
            time_seconds=2.0,
        ),
    )
    rows = _section_rows(contract)
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]

    assert contract.title == "Event Take"
    assert rows["take"] == "Take 2 (take_alt)"
    assert rows["playback state"] == "Active"
    assert rows["selected identity"] == "none"
    assert rows["playback target"] == "Active Kick / Take 2 (take_alt)"
    assert {
        "seek_here",
        "overwrite_main",
        "merge_main",
        "delete_take",
        "selection.select_every_other",
        "selection.renumber_cues_from_one",
    } <= set(action_ids)
    assert "send_selected_events_to_ma3" in action_ids
    assert "send_layer_to_ma3" not in action_ids
    assert "send_to_different_track_once" not in action_ids


def test_inspector_contract_layer_hit_uses_selected_layers_batch_scope_when_multiselect_is_active():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_empty")
    presentation.selected_layer_ids = [LayerId("layer_kick"), LayerId("layer_empty")]

    contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(
            kind="layer",
            layer_id=LayerId("layer_kick"),
            time_seconds=1.0,
        ),
    )
    batch_actions = next(
        section for section in contract.context_sections if section.section_id == "event-batch"
    ).actions
    select_every_other = next(
        action for action in batch_actions if action.action_id == "selection.select_every_other"
    )

    assert select_every_other.label == "Select Every Other in Selected Layers"
    assert select_every_other.params["scope_mode"] == "selected_layers_main"


def test_inspector_contract_uses_region_batch_scope_when_region_selected():
    presentation = _contract_test_presentation()
    presentation.selected_region_id = RegionId("region_1")
    presentation.regions = [
        RegionPresentation(
            region_id=RegionId("region_1"),
            start=0.9,
            end=1.6,
            label="Verse",
        )
    ]

    contract = build_timeline_inspector_contract(presentation)
    batch_actions = next(
        section for section in contract.context_sections if section.section_id == "event-batch"
    ).actions
    select_every_other = next(
        action for action in batch_actions if action.action_id == "selection.select_every_other"
    )
    renumber = next(
        action for action in batch_actions if action.action_id == "selection.renumber_cues_from_one"
    )

    assert select_every_other.label == "Select Every Other in Region"
    assert select_every_other.params["scope_mode"] == "region"
    assert select_every_other.params["scope_region_id"] == "region_1"
    assert renumber.label == "Renumber Cues from 1 in Region"


def test_inspector_contract_no_takes_layer_state():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_empty")

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]

    assert contract.title == "Layer Empty"
    assert rows["main take"] == "none"
    assert rows["takes"] == "none"
    assert "overwrite_main" not in action_ids
    assert "merge_main" not in action_ids
    assert "delete_take" not in action_ids


def test_inspector_contract_render_text_tracks_selection_transition_sequence():
    presentation = _contract_test_presentation()

    timeline_contract = build_timeline_inspector_contract(presentation)

    presentation.selected_layer_id = LayerId("layer_kick")
    layer_contract = build_timeline_inspector_contract(presentation)

    presentation.selected_take_id = TakeId("take_main")
    presentation.selected_event_ids = [EventId("main_evt")]
    event_contract = build_timeline_inspector_contract(presentation)

    presentation.selected_layer_id = None
    presentation.selected_take_id = None
    presentation.selected_event_ids = []
    cleared_contract = build_timeline_inspector_contract(presentation)

    assert render_inspector_contract_text(timeline_contract) == "No timeline object selected."
    assert render_inspector_contract_text(layer_contract) == "\n".join(
        [
            "Layer Kick",
            "id: layer_kick",
            "kind: EVENT",
            "main take: take_main",
            "takes: 2",
            "status flags: none",
            "playback state: Set Active",
            "sync state: Off",
            "sync mapping: none",
            "selected identity: Layer Kick (layer_kick)",
            "playback target: none",
        ]
    )
    assert render_inspector_contract_text(event_contract) == "\n".join(
        [
            "Event Main",
            "id: main_evt",
            "start: 1.00s",
            "end: 1.50s",
            "duration: 0.50s",
            "layer: Kick",
            "take: Main take (take_main)",
            "playback state: Set Active",
            "sync state: Off",
            "sync mapping: none",
            "selected identity: Event Main (main_evt) on Kick / Main take (take_main)",
            "playback target: none",
        ]
    )
    assert render_inspector_contract_text(cleared_contract) == "No timeline object selected."


def test_inspector_contract_empty_state_with_playback_target_shows_target_separately():
    presentation = _contract_test_presentation()
    presentation.active_playback_layer_id = LayerId("layer_kick")
    presentation.active_playback_take_id = TakeId("take_main")

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)

    assert contract.identity is None
    assert contract.title == "Timeline"
    assert rows["selected identity"] == "none"
    assert rows["playback target"] == "Active Kick / Main take (take_main)"
    assert render_inspector_contract_text(contract) == "\n".join(
        [
            "Timeline",
            "selected identity: none",
            "playback target: Active Kick / Main take (take_main)",
        ]
    )


def test_inspector_contract_no_takes_hit_target_excludes_take_actions():
    presentation = _contract_test_presentation()

    contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(
            kind="layer",
            layer_id=LayerId("layer_empty"),
            time_seconds=1.25,
        ),
    )
    section_ids = [section.section_id for section in contract.context_sections]
    action_ids = [
        action.action_id for section in contract.context_sections for action in section.actions
    ]

    assert contract.title == "Layer Empty"
    assert "take-actions" not in section_ids
    assert "overwrite_main" not in action_ids
    assert "merge_main" not in action_ids
    assert "delete_take" not in action_ids
