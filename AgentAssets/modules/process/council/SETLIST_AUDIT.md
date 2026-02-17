# Council Audit: Setlist Functionality

**Date:** December 2025  
**Proposal:** Audit existing setlist functionality for architectural integrity, clarity of purpose, and integration with core application

---

## Proposal Summary

**What exists:**
- Setlist entity and repository (scoped to projects)
- SetlistService for orchestrating batch processing
- SnapshotService for saving/restoring data states
- SetlistAudioInput block for audio entry point
- ExecutionStrategy value object for configurable processing
- Full UI integration (SetlistView, action config, error handling)
- Event-driven refresh when projects change

**Core Purpose:**
Enable users to process multiple audio files through the same project configuration, save results per song, and quickly switch between processed songs to compare results.

**Key Design Decisions:**
1. Setlists are part of projects (not separate entities)
2. Uses snapshot/checkpoint pattern (no execution engine changes)
3. Leverages existing quick_actions system for per-song configuration
4. Event-driven UI refresh for seamless integration

---

## Council Analysis

### Architect Analysis

**Problem Understanding:**
Setlist functionality allows batch processing of multiple audio files through a single project template. The implementation uses a snapshot pattern to save/restore data states without modifying the core execution engine.

**Key Concerns:**

**✅ Strengths:**
1. **Clean separation of concerns**: SetlistService orchestrates, SnapshotService handles persistence, execution engine unchanged
2. **Proper layer boundaries**: Domain entities (Setlist, SetlistSong), Application service (SetlistService), Infrastructure repositories
3. **Leverages existing patterns**: Uses ProjectService serialization, quick_actions system, event bus
4. **No execution engine changes**: Snapshot pattern means core app runs normally - aligns with "best part is no part"
5. **Project-scoped**: Setlists belong to projects, not global - correct abstraction level

**⚠️ Concerns:**
1. **SetlistAudioInput block**: Creates a special block type just for setlists. Could this be handled differently?
   - **Analysis**: Actually reasonable - provides clear entry point, avoids modifying LoadAudio. But worth questioning if LoadAudio could handle both cases.
2. **Snapshot storage**: Snapshots stored in database (via DataStateSnapshot entity). Could grow large.
   - **Analysis**: Acceptable for now, but should monitor. Could move to file-based storage later if needed.
3. **Circular dependency resolved**: SetlistService needs ApplicationFacade, but facade needs SetlistService. Currently resolved by setting facade after initialization.
   - **Analysis**: Works but fragile. Could use dependency injection or event-driven approach.

**Alternatives Considered:**
- **Alternative 1**: Modify LoadAudio to handle setlist context
  - **Why not**: Would complicate LoadAudio, violates single responsibility
- **Alternative 2**: Store snapshots as files instead of DB
  - **Why not**: DB is simpler for now, can migrate later if needed
- **Alternative 3**: Make setlists global instead of project-scoped
  - **Why not**: Would break mental model - setlists are about processing songs through a specific project configuration

**Vote: Approve with Conditions**

**Reasoning:**
The architecture is sound and follows established patterns. The snapshot approach is elegant and doesn't require execution engine changes. Setlists being project-scoped is correct. Minor concerns about SetlistAudioInput and snapshot storage are acceptable trade-offs.

**Conditions:**
1. Monitor snapshot storage size - consider file-based storage if DB grows too large
2. Document the circular dependency resolution pattern for future reference
3. Consider if LoadAudio could handle setlist context without SetlistAudioInput (future optimization)

---

### Systems Analysis

**Problem Understanding:**
Setlist functionality processes multiple songs sequentially, saves snapshots, and allows switching between them. Must handle errors gracefully, manage memory, and ensure stability.

**Key Concerns:**

**✅ Strengths:**
1. **Memory cleanup**: Block cleanup methods called after execution, garbage collection triggered
2. **Error handling**: Songs marked as failed with error messages, processing continues for other songs
3. **Snapshot isolation**: Each song's data stored separately, no conflicts
4. **Project verification**: Checks current project matches setlist's project before processing
5. **Event-driven refresh**: Uses BlockStatusChanged events for UI updates

**⚠️ Concerns:**
1. **Snapshot size**: Large projects with many data items could create large snapshots
   - **Analysis**: Monitor, but acceptable for now. Snapshots are only loaded when switching songs.
2. **Database growth**: Multiple setlists with many songs could grow database significantly
   - **Analysis**: Acceptable - databases can handle this. Can archive old setlists if needed.
3. **Memory during batch processing**: Processing many songs sequentially could accumulate memory
   - **Analysis**: Cleanup methods should handle this, but worth monitoring in production
4. **Error recovery**: If snapshot save fails, song processing succeeds but snapshot missing
   - **Analysis**: Should mark song as failed if snapshot save fails - current implementation may not handle this

**Failure Modes:**
1. **Snapshot save fails**: Song processed but snapshot not saved - user can't switch to that song
   - **Mitigation**: Should mark song as failed if snapshot save fails
2. **Project mismatch**: User switches projects, setlist becomes invalid
   - **Mitigation**: Already handled - setlist scoped to project, UI refreshes on project change
3. **Large snapshot restore**: Restoring very large snapshot could be slow
   - **Mitigation**: Acceptable - user-initiated action, can show progress

**Vote: Approve with Conditions**

**Reasoning:**
Error handling is robust, memory cleanup is in place, and the system handles edge cases well. Minor concerns about snapshot size and error recovery are acceptable.

**Conditions:**
1. Ensure snapshot save failures mark song as failed (verify current implementation)
2. Monitor snapshot sizes in production - consider compression or lazy loading if needed
3. Add progress indicator for large snapshot restores

---

### UX Analysis

**Problem Understanding:**
Users need to process multiple songs through their project, configure per-song settings, see progress, handle errors, and quickly switch between processed songs.

**Key Concerns:**

**✅ Strengths:**
1. **Clear mental model**: Setlists belong to projects - intuitive
2. **Integrated UI**: Setlist view is a tab alongside Node Editor - feels native
3. **Action configuration**: Uses existing quick_actions - familiar pattern
4. **Error visibility**: Error summary panel shows failed songs clearly
5. **Quick switching**: Dropdown and double-click to switch songs
6. **Auto-refresh**: UI updates automatically when switching songs

**⚠️ Concerns:**
1. **SetlistAudioInput requirement**: Users must add a special block for setlists
   - **Analysis**: Could be confusing - why not just use LoadAudio? But provides clear entry point.
2. **Action configuration complexity**: Configuring actions per song might be overwhelming
   - **Analysis**: Default actions help, but UI could be clearer about what actions do
3. **Execution strategy**: Three modes (full, actions_only, hybrid) - might be confusing
   - **Analysis**: Good to have options, but default should be simple (full execution)
4. **No visual feedback during snapshot restore**: When switching songs, no indication that restore is happening
   - **Analysis**: Should show loading indicator

**User Workflow:**
1. Create/open project ✅
2. Add blocks and connections ✅
3. Create setlist from folder ✅
4. Configure actions (optional) ⚠️ Could be clearer
5. Process songs ✅
6. Switch between songs ✅
7. View results ✅

**Discoverability:**
- Setlist view is a tab - discoverable ✅
- "Existing Setlists" section shows setlists for current project ✅
- Error messages are clear ✅

**Vote: Approve with Conditions**

**Reasoning:**
The UX is generally good - integrated, intuitive, and follows existing patterns. Minor improvements needed for clarity and feedback.

**Conditions:**
1. Add loading indicator when restoring snapshots (switching songs)
2. Improve action configuration UI - make it clearer what actions do and when to use them
3. Consider making SetlistAudioInput optional - allow LoadAudio to work in setlist context
4. Simplify execution strategy UI - make default clear, hide advanced options initially

---

### Pragmatic Analysis

**Problem Understanding:**
Setlist functionality is implemented and working. Need to assess implementation complexity, testing, and maintenance burden.

**Key Concerns:**

**✅ Strengths:**
1. **Reuses existing code**: SnapshotService uses ProjectService serialization, quick_actions system already exists
2. **Clean separation**: SetlistService orchestrates, repositories handle persistence, services handle business logic
3. **Testable**: Services are testable with mock repositories
4. **Event-driven**: Uses event bus for loose coupling
5. **Database migrations**: Handles schema changes gracefully

**⚠️ Concerns:**
1. **Code volume**: ~2000+ LOC for setlist functionality
   - **Analysis**: Reasonable for the feature scope - includes UI, service, repositories, entities
2. **Testing coverage**: Need to verify test coverage for setlist functionality
   - **Analysis**: Should have unit tests for SetlistService, integration tests for UI
3. **Maintenance burden**: Setlist functionality adds complexity to codebase
   - **Analysis**: Acceptable - well-structured, follows patterns, should be maintainable
4. **Circular dependency**: SetlistService needs ApplicationFacade, resolved manually
   - **Analysis**: Works but fragile - could use dependency injection pattern

**Implementation Quality:**
- **Code organization**: ✅ Well-structured, follows patterns
- **Error handling**: ✅ Comprehensive
- **Documentation**: ⚠️ Could use more inline docs
- **Type hints**: ✅ Present
- **Logging**: ✅ Good logging throughout

**Testing Strategy:**
- Unit tests: SetlistService, SnapshotService, repositories
- Integration tests: Full setlist workflow (create, process, switch)
- UI tests: SetlistView interactions

**Vote: Approve with Conditions**

**Reasoning:**
Implementation is solid, follows established patterns, and reuses existing code well. Code volume is reasonable for feature scope. Minor concerns about testing and documentation.

**Conditions:**
1. Add unit tests for SetlistService (if not already present)
2. Add integration tests for full setlist workflow
3. Improve inline documentation for complex methods
4. Consider dependency injection pattern for circular dependency resolution

---

## Unanimous Recommendation

**RECOMMENDATION: Approve with Conditions**

The Council unanimously approves the setlist functionality with minor improvements needed.

### Key Strengths

**Architect:**
- Clean separation of concerns, proper layer boundaries
- Snapshot pattern elegantly avoids execution engine changes
- Project-scoped design is correct abstraction level

**Systems:**
- Robust error handling and memory cleanup
- Graceful failure recovery
- Event-driven architecture for loose coupling

**UX:**
- Integrated seamlessly into application
- Clear mental model (setlists belong to projects)
- Uses familiar patterns (quick_actions, event bus)

**Pragmatic:**
- Reuses existing code effectively
- Well-structured and maintainable
- Reasonable code volume for feature scope

### Required Modifications

1. **Error Handling** (Systems):
   - Ensure snapshot save failures mark song as failed
   - Add progress indicator for large snapshot restores

2. **User Experience** (UX):
   - Add loading indicator when restoring snapshots (switching songs)
   - Improve action configuration UI clarity
   - Consider making SetlistAudioInput optional (allow LoadAudio in setlist context)

3. **Documentation** (Pragmatic):
   - Add unit tests for SetlistService
   - Add integration tests for full workflow
   - Improve inline documentation for complex methods

4. **Architecture** (Architect):
   - Monitor snapshot storage size - consider file-based storage if DB grows too large
   - Document circular dependency resolution pattern

### Alignment with Core Values

**"Best Part is No Part":**
✅ No execution engine changes - snapshot pattern means core app runs normally  
✅ Reuses existing serialization, quick_actions, event bus  
✅ Setlists are part of projects, not separate global entities

**"Simplicity and Refinement":**
✅ Simple mental model: process songs, save results, switch between them  
✅ Uses existing patterns instead of creating new ones  
⚠️ Could simplify execution strategy UI (hide advanced options initially)

### Success Criteria

The setlist functionality successfully:
1. ✅ Processes multiple songs through same project configuration
2. ✅ Saves/restores data states without modifying execution engine
3. ✅ Integrates seamlessly with existing application
4. ✅ Handles errors gracefully
5. ✅ Provides clear UI for configuration and management

### Next Steps

1. Verify snapshot save failure handling marks songs as failed
2. Add loading indicator for snapshot restore
3. Improve action configuration UI clarity
4. Add comprehensive tests (unit + integration)
5. Monitor snapshot storage size in production

---

## Conclusion

The setlist functionality is **well-designed, properly integrated, and aligns with core values**. The snapshot pattern is elegant and avoids execution engine changes. Minor improvements needed for error handling, UX feedback, and testing, but the core implementation is sound.

**Status: Approved for Production with Minor Enhancements**

