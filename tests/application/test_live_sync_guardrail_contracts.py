from __future__ import annotations

import pytest

from echozero.application.shared.ids import LayerId
from echozero.application.sync.models import LiveSyncState, SyncState
from echozero.application.timeline.intents import (
    ClearLayerLiveSyncPauseReason,
    DisableExperimentalLiveSync,
    EnableExperimentalLiveSync,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
)
from echozero.application.timeline.models import LayerSyncState


def test_sync_state_defaults_experimental_live_sync_disabled():
    state = SyncState()

    assert state.experimental_live_sync_enabled is False


def test_layer_sync_state_defaults_guardrail_fields():
    state = LayerSyncState()

    assert state.live_sync_state is LiveSyncState.OFF
    assert state.live_sync_pause_reason is None
    assert state.live_sync_divergent is False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (LiveSyncState.OFF, LiveSyncState.OFF),
        ("observe", LiveSyncState.OBSERVE),
        ("armed_write", LiveSyncState.ARMED_WRITE),
        ("paused", LiveSyncState.PAUSED),
    ],
)
def test_layer_sync_state_accepts_supported_live_sync_values(raw_value, expected):
    state = LayerSyncState(live_sync_state=raw_value)

    assert state.live_sync_state is expected


def test_layer_sync_state_rejects_unknown_live_sync_state():
    with pytest.raises(ValueError, match="live_sync_state must be one of: off, observe, armed_write, paused"):
        LayerSyncState(live_sync_state="invalid")


def test_layer_sync_state_normalizes_blank_pause_reason_to_none():
    state = LayerSyncState(live_sync_pause_reason="   ")

    assert state.live_sync_pause_reason is None


def test_experimental_live_sync_toggle_intents_are_constructible():
    assert isinstance(EnableExperimentalLiveSync(), EnableExperimentalLiveSync)
    assert isinstance(DisableExperimentalLiveSync(), DisableExperimentalLiveSync)


def test_set_layer_live_sync_state_keeps_layer_and_normalized_state():
    intent = SetLayerLiveSyncState(
        layer_id=LayerId("layer_live_sync"),
        live_sync_state="observe",
    )

    assert intent.layer_id == LayerId("layer_live_sync")
    assert intent.live_sync_state is LiveSyncState.OBSERVE


@pytest.mark.parametrize("layer_id", [None, "", "   "])
def test_set_layer_live_sync_state_requires_non_empty_layer_id(layer_id):
    with pytest.raises(ValueError, match="SetLayerLiveSyncState requires a non-empty layer_id"):
        SetLayerLiveSyncState(layer_id=layer_id, live_sync_state=LiveSyncState.OFF)


def test_set_layer_live_sync_state_rejects_unknown_state():
    with pytest.raises(ValueError, match="live_sync_state must be one of: off, observe, armed_write, paused"):
        SetLayerLiveSyncState(
            layer_id=LayerId("layer_live_sync"),
            live_sync_state="invalid",
        )


def test_set_layer_live_sync_pause_reason_trims_whitespace():
    intent = SetLayerLiveSyncPauseReason(
        layer_id=LayerId("layer_live_sync"),
        pause_reason="  drift detected  ",
    )

    assert intent.pause_reason == "drift detected"


@pytest.mark.parametrize("pause_reason", ["", "   "])
def test_set_layer_live_sync_pause_reason_requires_non_empty_value(pause_reason):
    with pytest.raises(ValueError, match="SetLayerLiveSyncPauseReason requires a non-empty pause_reason"):
        SetLayerLiveSyncPauseReason(
            layer_id=LayerId("layer_live_sync"),
            pause_reason=pause_reason,
        )


@pytest.mark.parametrize("layer_id", [None, "", "   "])
def test_clear_layer_live_sync_pause_reason_requires_non_empty_layer_id(layer_id):
    with pytest.raises(ValueError, match="ClearLayerLiveSyncPauseReason requires a non-empty layer_id"):
        ClearLayerLiveSyncPauseReason(layer_id=layer_id)

