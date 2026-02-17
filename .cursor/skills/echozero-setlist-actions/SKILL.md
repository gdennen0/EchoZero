---
name: echozero-setlist-actions
description: Integrate blocks into setlists, action sets, and commands in EchoZero. Use when adding block actions to setlists, configuring action items, quick actions for setlist processing, or when the user asks about setlist actions, action sets, or block commands.
---

# Setlist and Block Actions

## Action Flow

Setlist processing: Song -> Action pipeline -> Execute actions in order -> Project execution

## ActionItem Entity

`src/features/projects/domain/action_set.py`

```python
@dataclass
class ActionItem:
    action_type: str  # "block" or "project"
    action_name: str
    block_id: Optional[str]  # Required for block actions
    block_name: str
    action_args: Dict[str, Any]
    order_index: int
```

## Quick Actions

Register block actions for UI and setlist use:

```python
from src.application.blocks.quick_actions import quick_action, ActionCategory

@quick_action(
    "MyBlock",
    "My Action",
    description="Description",
    category=ActionCategory.CONFIGURE,
    icon="icon-name",
    primary=True
)
def my_action(facade, block_id: str, value=None, **kwargs):
    if value is not None:
        # Execute - update settings, etc.
        return {"success": True}
    # Return input spec for dialog
    return {"needs_input": True, "input_type": "choice", ...}
```

## Action Categories

`ActionCategory.EXECUTE`, `CONFIGURE`, `FILE`, `EDIT`, `VIEW`, `EXPORT`

## Setlist Processing

- SetlistService: `src/features/setlists/application/setlist_service.py`
- SetlistProcessingService: executes actions per song
- Block actions: resolved by block_id + action_name, runs quick action handler
- Project actions: Execute project, etc.

## Key: SetlistAudioInput

Setlist songs need audio. SetlistAudioInput block's "Set Audio File" action sets `audio_path` in metadata - required for setlist processing.

## Action Overrides

SetlistSong has `action_overrides: Dict[str, Dict[str, Any]]` - per-song overrides:
`{block_id: {action_name: action_args, ...}, ...}`

## Adding Block to Setlist Pipeline

1. Register quick action with `@quick_action(block_type, action_name, ...)`
2. Action appears in Action Set Editor
3. User adds to default actions or per-song overrides
4. SetlistProcessingService executes via block_id + action_name lookup

## Key Files

- Quick actions: `src/application/blocks/quick_actions.py`
- ActionItem/ActionSet: `src/features/projects/domain/action_set.py`
- SetlistProcessingService: `src/features/setlists/application/setlist_processing_service.py`
- Action Set Editor: `ui/qt_gui/views/action_set_editor.py`
