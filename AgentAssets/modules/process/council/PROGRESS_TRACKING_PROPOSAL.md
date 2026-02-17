# Council Decision: Centralized Progress Tracking System

## Proposal

Implement a centralized progress tracking system that provides very verbose, data-rich progress information. Start with a trial implementation for setlist "Process all" functionality, then expand application-wide.

## Problem

Current progress tracking is basic and lacks detailed information:
- No timing information (elapsed, estimated remaining)
- No performance metrics (CPU, memory usage)
- Limited error context
- No hierarchical view (setlist → song → action → block)
- Not query-able from multiple endpoints

Users need more visibility into what's happening during long-running operations.

## Evidence

- User feedback requesting more detailed progress information
- Debugging failures is difficult without context
- No way to estimate time remaining
- Current dialog shows minimal information

## Proposed Solution

**Approach 1: Event-Based Progress Store** (Recommended for MVP)

1. Create `ProgressEventStore` (singleton) that accumulates progress state
2. Extend event system with verbose `SetlistProgressEvent`
3. Update `SetlistService` to emit rich progress events
4. Update `SetlistProcessingDialog` to query store for detailed information
5. Provide query API for CLI/other endpoints

**Key Components:**
- `ProgressState` / `ProgressLevel` models for hierarchical progress
- `ProgressEventStore` for centralized state management
- Rich progress events with timing, metrics, error context
- Query methods for accessing current state and history

## Alternatives Considered

1. **Progress Context Manager**: Clean API but requires refactoring
2. **Progress Reporter Pattern**: Flexible but verbose to use
3. **Hybrid Approach**: Best long-term but more complex for MVP

**Why Approach 1?**
- Minimal changes to existing code
- Leverages existing event system
- Query-able from anywhere
- Can start small (just setlist)
- Scales naturally

## Scope

### Phase 1: MVP (Setlist Only)
- Core models and store
- Setlist integration
- Enhanced dialog
- Testing

**Estimated Time: 6-10 hours**

### Phase 2: Block Execution (Future)
- Extend to block execution
- Integrate with ExecutionEngine
- Block-level progress in UI

### Phase 3: Application-Wide (Future)
- Generic progress API
- CLI commands
- Historical tracking
- Performance metrics

## Costs

- **LOC**: ~500-800 lines (models, store, integration)
- **Dependencies**: None (uses existing event system)
- **Testing**: Unit tests for store, integration tests for setlist
- **Maintenance**: Low (event-based, self-contained)
- **Performance**: Minimal (in-memory store, event-driven)

## Benefits

- **User Experience**: Much better visibility into progress
- **Debugging**: Rich error context and timing information
- **Planning**: Time estimates help users plan
- **Monitoring**: Performance metrics for optimization
- **Extensibility**: Foundation for application-wide progress tracking

## Alignment with Core Values

### "Best Part is No Part"
- ✅ Reuses existing event system (no new infrastructure)
- ✅ Additive (doesn't break existing code)
- ✅ Questioned necessity: Real user need for better progress visibility

### "Simplicity and Refinement"
- ✅ Simple event-based approach
- ✅ Start small (setlist only), expand later
- ✅ Clear, single responsibility (progress tracking)

## Council Analysis Required

Please analyze from your lens:

### Architect
- Does this respect layer boundaries?
- What new dependencies does this create?
- Is this the right abstraction level?
- Does this fit existing patterns?

### Systems
- What failure modes does this introduce?
- Resource usage (memory for state storage)?
- Performance impact of event processing?
- Error handling strategy?

### UX
- Does this make user's job easier?
- How to display verbose info without overwhelming?
- Is it discoverable?
- Clear, actionable error messages?

### Pragmatic
- How hard is this to implement?
- Testing complexity?
- Scope manageable?
- Maintenance burden?

## Questions for Council

1. **Storage**: Should progress state persist across restarts? (Recommendation: No for MVP)
2. **History**: How much history to keep? (Recommendation: 100 operations)
3. **Performance**: Update frequency? (Recommendation: Every event, but throttle UI updates)
4. **UI**: How to display verbose info? (Recommendation: Expandable sections, tabs)
5. **Metrics**: What metrics to track? (Recommendation: Timing, basic CPU/memory for MVP)

## Recommendation

Proceed with **Approach 1 (Event-Based Progress Store)** for MVP, starting with setlist processing only.

**Rationale:**
- Solves real user problem (better progress visibility)
- Minimal complexity (leverages existing event system)
- Start small, expand later (aligns with core values)
- Low risk (additive, non-breaking)
- Clear path to application-wide expansion

**Success Criteria:**
- Setlist processing shows verbose progress (timing, metrics, errors)
- Progress state query-able from dialog and CLI
- No performance degradation
- Users report better visibility into progress

## Next Steps

1. Council review and feedback
2. Select approach (or propose alternative)
3. Create detailed API design
4. Implement MVP (setlist only)
5. Test and gather feedback
6. Expand to block execution
7. Expand application-wide

