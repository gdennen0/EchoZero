# Action Item Editor Refactor - Robust Implementation

## Problem Statement

The current ActionSetEditor has several critical issues:
1. **Glitchy behavior**: Actions don't save properly when values change
2. **Order corruption**: Deleting actions changes the order of remaining items
3. **No undo/redo**: Changes aren't tracked in the command system
4. **Inconsistent patterns**: Doesn't follow the same robust patterns used for blocks

## Solution: Command-Based Architecture

Refactor ActionSetEditor to use the same command-based architecture as blocks:

### Core Principles

1. **All operations go through CommandBus** - No direct database modifications
2. **Track by ID, not row index** - Use action item IDs for reliable tracking
3. **Maintain order_index** - Sort by order_index, update indices on changes
4. **Immediate persistence** - Commands save to database automatically
5. **Undo/redo support** - All operations are undoable
6. **UI refresh on undo stack change** - Follow block panel pattern

### Implementation Plan

#### 1. Command Integration

- Use `AddActionItemCommand` for new actions
- Use `UpdateActionItemCommand` for modifications
- Use `DeleteActionItemCommand` for deletions
- Use `ReorderActionItemsCommand` for order changes

#### 2. Order Preservation

- Always sort actions by `order_index` when displaying
- Update `order_index` for all affected items when deleting
- Use `ReorderActionItemsCommand` when order changes

#### 3. ID-Based Tracking

- Store action item IDs in table widget user data
- Use IDs to find actions instead of row indices
- Prevent stale references after refresh

#### 4. UI Refresh Pattern

- Subscribe to undo stack changes
- Refresh table when commands complete
- Preserve selection by ID after refresh

#### 5. Immediate Persistence

- Commands automatically save to database
- No separate "save" step needed
- Auto-save on every change (via commands)

### Code Changes

#### Key Methods to Refactor

1. `_refresh_actions_list()` - Sort by order_index, use IDs
2. `_on_block_changed_inline()` - Use UpdateActionItemCommand
3. `_on_action_changed_inline()` - Use UpdateActionItemCommand
4. `_on_delete_action()` - Use DeleteActionItemCommand + ReorderActionItemsCommand
5. `_on_edit_params()` - Use UpdateActionItemCommand
6. `_on_save_set()` - Remove manual save, rely on commands

#### New Methods

1. `_get_action_by_id()` - Find action by ID
2. `_refresh_from_database()` - Reload from DB after command
3. `_update_order_indices()` - Recalculate order_index after changes
4. `_subscribe_to_undo_stack()` - Refresh on undo/redo

### Testing Checklist

- [ ] Add action - persists immediately
- [ ] Update action - persists immediately
- [ ] Delete action - order preserved
- [ ] Reorder actions - order persists
- [ ] Undo add - action removed
- [ ] Undo delete - action restored with correct order
- [ ] Undo update - old values restored
- [ ] Redo operations - all work correctly
- [ ] Multiple rapid changes - no glitches
- [ ] Project reload - actions load correctly

### Success Criteria

1. All operations use commands
2. Order is always preserved
3. Changes persist immediately
4. Undo/redo works for all operations
5. No glitchy behavior
6. Follows same patterns as blocks


