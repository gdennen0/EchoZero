"""Explicit timeline intents for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
)
from echozero.application.shared.ids import LayerId, TakeId, EventId
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import LiveSyncState, coerce_live_sync_state


class TimelineIntent:
    """Marker base type for timeline intents."""


@dataclass(slots=True)
class SelectLayer(TimelineIntent):
    layer_id: LayerId | None


@dataclass(slots=True)
class SelectTake(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None


@dataclass(slots=True)
class SelectEvent(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None
    event_id: EventId | None
    mode: str = "replace"


@dataclass(slots=True)
class ClearSelection(TimelineIntent):
    pass


@dataclass(slots=True)
class SelectAllEvents(TimelineIntent):
    pass


@dataclass(slots=True)
class ToggleLayerExpanded(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class TriggerTakeAction(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId
    action_id: str


@dataclass(slots=True)
class MoveEvent(TimelineIntent):
    event_id: EventId
    new_start: float


@dataclass(slots=True)
class MoveSelectedEvents(TimelineIntent):
    delta_seconds: float
    target_layer_id: LayerId | None = None


@dataclass(slots=True)
class TrimEvent(TimelineIntent):
    event_id: EventId
    new_range: TimeRange


@dataclass(slots=True)
class NudgeSelectedEvents(TimelineIntent):
    direction: int
    steps: int = 1


@dataclass(slots=True)
class DuplicateSelectedEvents(TimelineIntent):
    steps: int = 1


@dataclass(slots=True)
class Play(TimelineIntent):
    pass


@dataclass(slots=True)
class Pause(TimelineIntent):
    pass


@dataclass(slots=True)
class Stop(TimelineIntent):
    pass


@dataclass(slots=True)
class Seek(TimelineIntent):
    position: float


@dataclass(slots=True)
class ToggleMute(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class ToggleSolo(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class SetGain(TimelineIntent):
    layer_id: LayerId
    gain_db: float


@dataclass(slots=True)
class EnableSync(TimelineIntent):
    mode: SyncMode


@dataclass(slots=True)
class DisableSync(TimelineIntent):
    pass


@dataclass(slots=True)
class EnableExperimentalLiveSync(TimelineIntent):
    pass


@dataclass(slots=True)
class DisableExperimentalLiveSync(TimelineIntent):
    pass


@dataclass(slots=True)
class SetLayerLiveSyncState(TimelineIntent):
    layer_id: LayerId
    live_sync_state: LiveSyncState | str

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerLiveSyncState requires a non-empty layer_id")
        self.live_sync_state = coerce_live_sync_state(self.live_sync_state)


@dataclass(slots=True)
class SetLayerLiveSyncPauseReason(TimelineIntent):
    layer_id: LayerId
    pause_reason: str

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerLiveSyncPauseReason requires a non-empty layer_id")
        reason = self.pause_reason.strip()
        if not reason:
            raise ValueError("SetLayerLiveSyncPauseReason requires a non-empty pause_reason")
        self.pause_reason = reason


@dataclass(slots=True)
class ClearLayerLiveSyncPauseReason(TimelineIntent):
    layer_id: LayerId

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("ClearLayerLiveSyncPauseReason requires a non-empty layer_id")


@dataclass(slots=True)
class OpenPushToMA3Dialog(TimelineIntent):
    selection_event_ids: list[EventId]


@dataclass(slots=True)
class SetPushTrackOptions(TimelineIntent):
    tracks: list[ManualPushTrackOption]


@dataclass(slots=True)
class SelectPushTargetTrack(TimelineIntent):
    target_track_coord: str
    layer_id: LayerId | None = None

    def __post_init__(self) -> None:
        if not self.target_track_coord or not self.target_track_coord.strip():
            raise ValueError("SelectPushTargetTrack requires a non-empty target_track_coord")


@dataclass(slots=True)
class ExitPushToMA3Mode(TimelineIntent):
    pass


@dataclass(slots=True)
class ConfirmPushToMA3(TimelineIntent):
    target_track_coord: str
    selected_event_ids: list[EventId]

    def __post_init__(self) -> None:
        if not self.target_track_coord or not self.target_track_coord.strip():
            raise ValueError("ConfirmPushToMA3 requires a non-empty target_track_coord")
        if not self.selected_event_ids:
            raise ValueError("ConfirmPushToMA3 requires at least one selected_event_id")


@dataclass(slots=True)
class OpenPullFromMA3Dialog(TimelineIntent):
    pass


@dataclass(slots=True)
class ExitPullFromMA3Workspace(TimelineIntent):
    pass


@dataclass(slots=True)
class SelectPullSourceTracks(TimelineIntent):
    source_track_coords: list[str]

    def __post_init__(self) -> None:
        if not self.source_track_coords:
            raise ValueError("SelectPullSourceTracks requires at least one source_track_coord")
        if not all(coord and str(coord).strip() for coord in self.source_track_coords):
            raise ValueError("SelectPullSourceTracks requires non-empty source_track_coords")


@dataclass(slots=True)
class SetPullTrackOptions(TimelineIntent):
    tracks: list[ManualPullTrackOption]


@dataclass(slots=True)
class SelectPullSourceTrack(TimelineIntent):
    source_track_coord: str

    def __post_init__(self) -> None:
        if not self.source_track_coord or not self.source_track_coord.strip():
            raise ValueError("SelectPullSourceTrack requires a non-empty source_track_coord")


@dataclass(slots=True)
class SetPullSourceEvents(TimelineIntent):
    events: list[ManualPullEventOption]


@dataclass(slots=True)
class SelectPullSourceEvents(TimelineIntent):
    selected_ma3_event_ids: list[str]

    def __post_init__(self) -> None:
        if not self.selected_ma3_event_ids:
            raise ValueError("SelectPullSourceEvents requires at least one selected_ma3_event_id")
        if not all(event_id and str(event_id).strip() for event_id in self.selected_ma3_event_ids):
            raise ValueError("SelectPullSourceEvents requires non-empty selected_ma3_event_ids")


@dataclass(slots=True)
class SelectPullTargetLayer(TimelineIntent):
    target_layer_id: LayerId

    def __post_init__(self) -> None:
        if self.target_layer_id is None or not str(self.target_layer_id).strip():
            raise ValueError("SelectPullTargetLayer requires a non-empty target_layer_id")


@dataclass(slots=True)
class ConfirmPullFromMA3(TimelineIntent):
    source_track_coord: str
    selected_ma3_event_ids: list[str]
    target_layer_id: LayerId
    import_mode: str = "new_take"

    def __post_init__(self) -> None:
        if not self.source_track_coord or not self.source_track_coord.strip():
            raise ValueError("ConfirmPullFromMA3 requires a non-empty source_track_coord")
        if not self.selected_ma3_event_ids:
            raise ValueError("ConfirmPullFromMA3 requires at least one selected_ma3_event_id")
        if not all(event_id and str(event_id).strip() for event_id in self.selected_ma3_event_ids):
            raise ValueError("ConfirmPullFromMA3 requires non-empty selected_ma3_event_ids")
        if self.target_layer_id is None or not str(self.target_layer_id).strip():
            raise ValueError("ConfirmPullFromMA3 requires a non-empty target_layer_id")


@dataclass(slots=True)
class ApplyPullFromMA3(TimelineIntent):
    pass


@dataclass(slots=True)
class PreviewTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("PreviewTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id


@dataclass(slots=True)
class ApplyTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("ApplyTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id


@dataclass(slots=True)
class CancelTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("CancelTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id
