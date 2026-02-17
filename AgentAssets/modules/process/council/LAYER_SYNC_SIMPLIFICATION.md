# Layer Sync Simplification - Council Decision

## Proposal

Simplify ShowManager layer sync system from complex controller-based architecture to pure event-driven translation manager, while keeping auto-discovery for UI dialogs.

## Council Analysis

### Architect Analysis

**Structural Concerns:**
- Current `LayerSyncController` introduces unnecessary abstraction layer
- Complex mapping dictionaries create coupling between MA3 and Editor representations
- Manual sync buttons violate single-responsibility (UI should reflect state, not trigger operations)

**Alternatives Considered:**
1. Keep controller but simplify - Still adds abstraction overhead
2. Full removal of auto-discovery - User feedback suggests it's useful for dialogs
3. Hybrid approach (proposed) - Auto-discovery for dialogs, event-driven for sync - **BEST**

**Structural Benefits:**
- Clear separation: Discovery (read-only, UI) vs Sync (reactive, automatic)
- Event-driven architecture aligns with existing Editor/MA3 event patterns
- Removes mapping dictionaries that duplicate information already in `synced_layers`

**Vote:** ✅ **Approve** - Architecture is cleaner, boundaries are clearer

---

### Systems Engineer Analysis

**Infrastructure Concerns:**
- Auto-discovery polling could cause unnecessary OSC traffic if done too frequently
- Event handlers must be thread-safe (OSC listener runs on separate thread)
- Need to prevent infinite sync loops (Editor change → MA3 update → OSC event → Editor change...)

**Failure Modes:**
- MA3 disconnects → Should handle gracefully, queue updates
- Editor block deleted → Should clean up synced layers
- OSC message loss → Need idempotent sync logic

**Performance:**
- Event-driven approach more efficient than polling
- Auto-discovery can be throttled/debounced
- Sync operations should be batched where possible

**Vote:** ✅ **Approve with Conditions**
- Use logic-based triggers for auto-discovery (event-driven, not timer-based)
  - Fetch when: Dialog opens, MA3 listener starts, structure changes detected, settings loaded
  - Skip when: MA3 not ready, already fetching, data unchanged (signature check)
- Add sync loop prevention (track recent syncs, ignore duplicates)
- Ensure thread-safe event handlers

---

### UX Engineer Analysis

**User Impact:**
- Removing manual sync buttons reduces cognitive load
- Auto-discovery in dialogs improves discoverability
- Automatic sync aligns with user mental model ("things just work")

**Workflow Improvements:**
1. User adds MA3 track → Editor layer auto-created → Events auto-synced
2. User adds Editor layer → MA3 track auto-created → Events auto-synced
3. Changes propagate automatically (no "sync now" needed)

**Interface Clarity:**
- Simple list shows: Layer name, Track coord, Status (synced/error), Enable toggle
- No confusing "mapping table" with discovered vs synced items
- Clear indication of what's synced vs what's available

**Potential Issues:**
- Users might want to preview before syncing (keep dialogs for explicit selection)
- Need clear error messages when sync fails
- Need visual feedback during sync operations

**Vote:** ✅ **Approve** - Improves UX significantly, removes friction

---

### Pragmatic Engineer Analysis

**Implementation Complexity:**
- Removing controller: Low complexity (move logic to event handlers)
- Removing mapping dictionaries: Medium complexity (30+ references to clean up)
- Adding event handlers: Low complexity (clear patterns exist)
- Keeping auto-discovery: Low complexity (already exists, just clarify scope)

**Testing:**
- Event handlers easier to test than controller with internal state
- Can mock OSC listener and event bus for unit tests
- Integration tests: Add layer → verify sync → verify MA3 update

**Maintenance Burden:**
- Fewer moving parts = easier to debug
- Clear event flow = easier to trace issues
- Less state to manage = fewer bugs

**Delivery Approach:**
1. Fix immediate bugs (remove `layer_mappings` references)
2. Simplify settings (already done in previous refactor)
3. Replace controller with event handlers (incremental)
4. Remove manual buttons (UI cleanup)
5. Add sync loop prevention (safety)

**Risk Assessment:**
- Medium risk: Breaking existing synced layers during migration
- Mitigation: Ensure `synced_layers` format is backward compatible

**Vote:** ✅ **Approve with Conditions**
- Implement incrementally (don't remove old code until new code works)
- Add comprehensive logging for sync operations
- Test with existing projects to ensure migration works

---

## Unanimous Recommendation

**✅ Proceed with Modifications**

The council unanimously approves the simplification, with the following clarifications and conditions:

### Core Decision

1. **Keep auto-discovery** - But only for UI dialogs (populating "Add" dialogs), not for automatic sync
2. **Remove manual sync buttons** - Sync should be automatic and reactive
3. **Replace/simplify LayerSyncController** - Use direct event handlers instead
4. **Remove mapping dictionaries** - Use `synced_layers` as single source of truth

### Implementation Conditions

1. **Logic-based auto-discovery triggers** - No timers/debouncing, only fetch when:
   - Dialog opens (user action)
   - MA3 listener starts (connection event)
   - MA3 structure changes (signature comparison)
   - Settings loaded (initialization)
   - Skip if: MA3 not ready, already fetching, data unchanged
2. **Add sync loop prevention** - Track recent syncs, ignore duplicate/cyclic updates
3. **Thread-safe event handlers** - Ensure OSC listener thread safely communicates with main thread
4. **Incremental migration** - Don't remove old code until new code is tested and working
5. **Comprehensive logging** - Add clear log messages for sync operations (when, what, why)
6. **Backward compatibility** - Ensure existing `synced_layers` data continues to work

### Scope Clarification

**Auto-Discovery Usage:**
- ✅ Used to populate "Add Editor Layer" dialog with available Editor layers
- ✅ Used to populate "Add MA3 Track" dialog with available MA3 tracks
- ❌ NOT used to automatically create sync relationships
- ❌ NOT used to trigger sync operations

**Sync Operations:**
- ✅ Automatic when synced layers change on either side
- ✅ Event-driven (OSC events or Editor events trigger sync)
- ❌ No manual "sync now" button
- ❌ No periodic polling for sync

### Expected Benefits

1. **Simpler architecture** - Fewer abstractions, clearer data flow
2. **Better UX** - Automatic sync, no manual buttons
3. **Easier debugging** - Event-driven flow is easier to trace
4. **Less code** - Remove ~500+ lines of controller/mapping logic
5. **Better maintainability** - Clear separation of concerns

---

## Action Items

- [ ] Fix immediate bugs (remove all `layer_mappings` references)
- [ ] Add debouncing to auto-discovery functions
- [ ] Implement event handlers for MA3 → Editor sync
- [ ] Implement event handlers for Editor → MA3 sync
- [ ] Add sync loop prevention logic
- [ ] Remove manual sync buttons from UI
- [ ] Simplify LayerSyncController or remove if unnecessary
- [ ] Add comprehensive logging for sync operations
- [ ] Test backward compatibility with existing projects
- [ ] Update documentation to reflect new architecture

---

**Council Vote:** ✅ Unanimous Approval (4/4)

**Date:** Current Session

**Next Review:** After implementation complete
