# MA3 Create Layer Feature

## Overview

Added the ability to create new EchoZero layers directly from unmapped MA3 tracks in the ShowManager panel's Layer Mapping table.

---

## Feature Description

When an MA3 track is unmapped (has no corresponding EZ layer), a **"➕ Create New Layer..."** option appears in the EZ Layer dropdown. This allows users to quickly create and map new EZ layers without leaving the ShowManager interface.

---

## User Workflow

### 1. Identify Unmapped Tracks
- Unmapped MA3 tracks are highlighted with a light red background (`#3d1f1f`)
- Status column shows "⚠ Unmapped"
- EZ Layer dropdown shows **"➕ Create New Layer..."** as the first option

### 2. Create Layer
1. Click the **EZ Layer dropdown** for an unmapped track
2. Select **"➕ Create New Layer..."** from the dropdown
3. A dialog appears with a suggested layer name based on the MA3 track name
4. Edit the layer name if desired
5. Click OK to create the layer

### 3. Name Generation
The system automatically generates a layer name by:
- Converting the MA3 track name to lowercase
- Replacing spaces and hyphens with underscores
- Removing special characters
- Adding "layer_" prefix if not present
- Adding numeric suffix if name already exists (e.g., `layer_kicks_1`)

**Examples:**
- MA3 track "Kick Drum" → `layer_kick_drum`
- MA3 track "Hi-Hats" → `layer_hi_hats`
- MA3 track "Snare (Main)" → `layer_snare_main`

### 4. Automatic Mapping
Once created, the layer is automatically mapped to the MA3 track, and the table updates to show:
- Status: "✓ Mapped" (green)
- EZ Layer dropdown shows the newly created layer
- "Delete" button appears in Actions column

---

## Implementation Details

### UI Changes
**File:** `ui/qt_gui/block_panels/show_manager_panel.py`

#### EZ Layer Dropdown Enhancement
- **Before:** Simple dropdown with existing layers only
- **After:** 
  - "➕ Create New Layer..." option for unmapped MA3 tracks
  - Separator line after create option
  - All existing layers below
  - Stores MA3 track info for create functionality

#### New Handler Method
```python
def _on_create_layer_from_ma3(self, track_info: MA3TrackInfo, ma3_coord: str):
    """
    Create a new EZ layer from an MA3 track and automatically map it.
    
    Features:
    - Smart name generation from MA3 track name
    - User confirmation dialog with editable name
    - Duplicate detection and handling
    - Automatic mapping creation
    - Local cache update
    """
```

### Name Sanitization
```python
import re
base_name = track_info.name.lower().strip()
base_name = re.sub(r'[^\w\s-]', '', base_name)  # Remove special chars
base_name = re.sub(r'[-\s]+', '_', base_name)   # Replace spaces/hyphens
```

### Duplicate Handling
If a layer with the same name already exists:
1. Prompt user: "Layer exists. Map to existing layer?"
2. **Yes:** Create mapping to existing layer
3. **No:** Cancel operation

---

## Current Limitations (Phase 4)

### Local Cache Only
- Layers are currently added to the local `_ez_layers` cache
- **Not yet created in the actual Editor block**
- A note is logged: "Layer created locally. In Phase 5, this will create the layer in the Editor block."

### Connected Editor Blocks
The "Refresh EZ Layers" button now queries connected Editor blocks:
1. Finds all Editor blocks connected via the `manipulator` port
2. Reads layer information from each Editor's UI state
3. Populates the dropdown with all available layers
4. Shows helpful messages if no Editor blocks are connected

**Connection Setup:**
- Connect ShowManager's `manipulator` port to Editor's `manipulator` port
- This creates a bidirectional communication channel
- ShowManager can now read layers and (in Phase 5) create new ones

### Phase 5 Integration (Planned)
In Phase 5, layer creation will be enhanced to:
1. Send a command to the Editor block via manipulator port
2. Actually create the layer in the Editor's data structure
3. Synchronize the layer across all connected Editor blocks
4. Persist the layer in the project file

---

## User Experience Benefits

### 1. Streamlined Workflow
- No need to switch between ShowManager and Editor
- Integrated into the EZ Layer dropdown for seamless UX
- Immediate visual feedback

### 2. Smart Defaults
- Intelligent name generation reduces manual typing
- Automatic conflict detection prevents errors
- Consistent naming conventions

### 3. Visual Clarity
- Clear distinction between mapped and unmapped tracks
- Color-coded status indicators
- Contextual action buttons

---

## Testing

### Manual Test Steps
1. **Load MA3 Structure:**
   - Click "Refresh MA3 Tracks" in ShowManager panel
   - Verify unmapped tracks show "Create Layer" button

2. **Create Layer:**
   - Click "Create Layer" for an unmapped track
   - Verify suggested name is reasonable
   - Edit name if desired, click OK
   - Verify layer appears in EZ Layer dropdown
   - Verify status changes to "✓ Mapped"

3. **Duplicate Handling:**
   - Try to create a layer with an existing name
   - Verify duplicate detection dialog appears
   - Test both "Yes" and "No" options

4. **Name Sanitization:**
   - Test with MA3 tracks containing:
     - Spaces: "Kick Drum" → `layer_kick_drum`
     - Special chars: "Hi-Hat (Main)" → `layer_hi_hat_main`
     - Hyphens: "Sub-Bass" → `layer_sub_bass`
     - Numbers: "Tom 1" → `layer_tom_1`

---

## Code Changes Summary

### Modified Files
1. **`ui/qt_gui/block_panels/show_manager_panel.py`**
   - Enhanced Actions column with conditional buttons
   - Added `_on_create_layer_from_ma3()` method
   - Added name sanitization logic
   - Added duplicate detection

### No New Files
All changes are contained within existing files.

---

## Future Enhancements (Phase 5)

1. **Editor Integration:**
   - Send `CreateLayerCommand` to Editor block
   - Receive confirmation of layer creation
   - Handle creation failures gracefully

2. **Batch Creation:**
   - "Create All Unmapped" button
   - Bulk layer creation with progress indicator

3. **Layer Templates:**
   - Predefined layer configurations (color, classification, etc.)
   - Apply template when creating layer

4. **Undo/Redo:**
   - Integrate with command bus for undo support
   - Revert layer creation if needed

---

## Related Documentation
- `MA3_LAYER_MAPPING_IMPLEMENTATION_SUMMARY.md` - Overall layer mapping feature
- `docs/encyclopedia/04-ui/show-manager-panel.md` - ShowManager UI guide
- `AgentAssets/modules/domains/editor/` - Editor block documentation

---

**Implementation Date:** January 11, 2026  
**Status:** Complete (Phase 4)  
**Next Steps:** Phase 5 - Editor block integration
