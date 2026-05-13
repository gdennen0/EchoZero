"""Clean-sheet MA3 show planning surface for EchoZero.
Exists to expose song/setlist/automator platform primitives as one app package.
Connects spreadsheet import, sequence planning, and MA3SongManager sync contracts.
"""

from echozero.application.ma3_show.models import (
    AutomatorEvent,
    EchoZeroShowPlan,
    MA3ExecutorPlacement,
    MA3MasterOutputAssignment,
    MA3ObjectMapping,
    MA3SongManagerSnapshot,
    SequenceBlockPolicy,
    Setlist,
    ShowSong,
    SongSection,
)
from echozero.application.ma3_show.planning import (
    allocate_song_sequence_mappings,
    build_automator_events,
    command_for_sequence_cue,
)
from echozero.application.ma3_show.event_adapter import automator_event_to_ma3_snapshot
from echozero.application.ma3_show.spreadsheet import (
    SpreadsheetImportResult,
    SpreadsheetSongRow,
    import_setlist_csv,
)
from echozero.application.ma3_show.sync_contract import (
    ShowPlanDiff,
    ShowPlanPreflight,
    build_show_plan,
    diff_show_plan_snapshot,
    preflight_show_plan,
)

__all__ = [
    "AutomatorEvent",
    "EchoZeroShowPlan",
    "MA3ExecutorPlacement",
    "MA3MasterOutputAssignment",
    "MA3ObjectMapping",
    "MA3SongManagerSnapshot",
    "SequenceBlockPolicy",
    "Setlist",
    "ShowSong",
    "SongSection",
    "SpreadsheetImportResult",
    "SpreadsheetSongRow",
    "ShowPlanDiff",
    "ShowPlanPreflight",
    "allocate_song_sequence_mappings",
    "automator_event_to_ma3_snapshot",
    "build_automator_events",
    "build_show_plan",
    "command_for_sequence_cue",
    "diff_show_plan_snapshot",
    "import_setlist_csv",
    "preflight_show_plan",
]
