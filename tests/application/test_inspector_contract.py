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
    TakeLanePresentation,
    TimelinePresentation,
    ManualPullEventOptionPresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.sync.models import LiveSyncState


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


def _section_rows(contract):
    return {row.label: row.value for section in contract.sections for row in section.rows}


def test_inspector_contract_no_selection_state():
    contract = build_timeline_inspector_contract(_contract_test_presentation())
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert contract.identity is None
    assert contract.title == "No timeline object selected."
    assert render_inspector_contract_text(contract) == "No timeline object selected."
    assert "pull_from_ma3" in action_ids


def test_inspector_contract_layer_selection_state():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.layers[0].sync_target_label = "tc1_tg2_tr3"
    presentation.batch_transfer_plan = BatchTransferPlanPresentation(
        plan_id="plan_123",
        operation_type="mixed",
        rows=[
            BatchTransferPlanRowPresentation(
                row_id="row_1",
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
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]
    section_ids = [section.section_id for section in contract.context_sections]

    assert contract.identity is not None
    assert contract.identity.object_type == "layer"
    assert contract.title == "Layer Kick"
    assert rows["id"] == "layer_kick"
    assert rows["kind"] == "EVENT"
    assert rows["main take"] == "take_main"
    assert rows["status flags"] == "none"
    assert rows["sync state"] == "Off"
    assert rows["sync mapping"] == "tc1_tg2_tr3"
    assert rows["transfer plan"] == "mixed plan_123 (ready 1, blocked 0, failed 0)"
    assert {"toggle_mute", "toggle_solo", "gain_down", "gain_unity", "gain_up"} <= set(action_ids)
    assert {"pull_from_ma3", "open_batch_transfer_workspace"} <= set(action_ids)
    assert "sync-transfer" in section_ids
    assert "live-sync" not in [section.section_id for section in contract.context_sections]


def test_inspector_contract_push_mode_layer_actions_and_facts():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.manual_push_flow.push_mode_active = True
    presentation.manual_push_flow.available_tracks = []
    presentation.layers[0].push_target_label = "Track 3 (tc1_tg2_tr3) - Bass"
    presentation.layers[0].push_selection_count = 1
    presentation.layers[0].push_row_status = "ready"
    presentation.batch_transfer_plan = BatchTransferPlanPresentation(
        plan_id="push:timeline_contract",
        operation_type="push",
        rows=[
            BatchTransferPlanRowPresentation(
                row_id="push:layer_kick",
                direction="push",
                source_label="Kick",
                target_label="Track 3 (tc1_tg2_tr3) - Bass",
                source_layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
                selected_event_ids=[EventId("main_evt")],
                selected_count=1,
                status="ready",
            )
        ],
        ready_count=1,
    )

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert rows["push mode"] == "active"
    assert rows["push target"] == "Track 3 (tc1_tg2_tr3) - Bass"
    assert rows["push selection"] == "1"
    assert rows["push row"] == "ready"
    assert {"select_push_target_track", "preview_push_diff", "exit_push_mode"} <= set(action_ids)


def test_inspector_contract_pull_workspace_actions_and_facts():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_kick")
    presentation.manual_pull_flow.workspace_active = True
    presentation.manual_pull_flow.available_tracks = []
    presentation.manual_pull_flow.active_source_track_coord = "tc1_tg2_tr5"
    presentation.manual_pull_flow.available_events = [
        ManualPullEventOptionPresentation(
            event_id="ma3_evt_1",
            label="Cue 1",
            start=1.0,
            end=1.5,
        )
    ]
    presentation.layers[0].pull_target_label = "Kick"
    presentation.layers[0].pull_selection_count = 2
    presentation.layers[0].pull_row_status = "ready"
    presentation.batch_transfer_plan = BatchTransferPlanPresentation(
        plan_id="pull:timeline_contract",
        operation_type="pull",
        rows=[
            BatchTransferPlanRowPresentation(
                row_id="pull:tc1_tg2_tr5",
                direction="pull",
                source_label="Track 5 (tc1_tg2_tr5)",
                target_label="Kick",
                source_track_coord="tc1_tg2_tr5",
                target_layer_id=LayerId("layer_kick"),
                selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"],
                selected_count=2,
                status="ready",
            )
        ],
        ready_count=1,
    )

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert rows["pull workspace"] == "active"
    assert rows["pull target"] == "Kick"
    assert rows["pull selection"] == "2"
    assert rows["pull row"] == "ready"
    assert {
        "select_pull_source_tracks",
        "select_pull_source_events",
        "set_pull_target_layer_mapping",
        "preview_pull_diff",
        "exit_pull_workspace",
    } <= set(action_ids)


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

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert contract.identity is not None
    assert contract.identity.object_type == "event"
    assert contract.title == "Event Main"
    assert rows["id"] == "main_evt"
    assert rows["start"] == "1.00s"
    assert rows["end"] == "1.50s"
    assert rows["duration"] == "0.50s"
    assert rows["take"] == "Main take (take_main)"
    assert {"push_to_ma3", "pull_from_ma3"} <= set(action_ids)


def test_inspector_contract_take_event_state():
    presentation = _contract_test_presentation()

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
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert contract.title == "Event Take"
    assert rows["take"] == "Take 2 (take_alt)"
    assert {"seek_here", "overwrite_main", "merge_main"} <= set(action_ids)


def test_inspector_contract_no_takes_layer_state():
    presentation = _contract_test_presentation()
    presentation.selected_layer_id = LayerId("layer_empty")

    contract = build_timeline_inspector_contract(presentation)
    rows = _section_rows(contract)
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert contract.title == "Layer Empty"
    assert rows["main take"] == "none"
    assert rows["takes"] == "none"
    assert "overwrite_main" not in action_ids
    assert "merge_main" not in action_ids


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
            "sync state: Off",
            "sync mapping: none",
            "transfer plan: none",
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
        ]
    )
    assert render_inspector_contract_text(cleared_contract) == "No timeline object selected."


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
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert contract.title == "Layer Empty"
    assert "take-actions" not in section_ids
    assert "overwrite_main" not in action_ids
    assert "merge_main" not in action_ids
