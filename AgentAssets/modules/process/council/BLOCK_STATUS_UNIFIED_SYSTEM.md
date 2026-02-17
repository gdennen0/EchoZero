# Council Decision: Unified Block Status System

**Proposal:** Unify block status system into single source of truth with automatic synchronization

**Date:** 2024

**Status:** Under Review

---

## Proposal Summary

**Problem:**
- Current system has two separate status services (BlockStatusService for ShowManager, DataStateService for others)
- Status dots in panels, nodes, and other UI components don't sync properly
- Not all blocks implement `get_status_levels()` - inconsistent status display
- Cache invalidation timing issues
- BlockStatusDot widget must choose which service to use based on block type

**Proposed Solution:**
1. Make BlockStatusService the single source of truth for ALL blocks
2. All blocks must implement `get_status_levels()` with data state checks included
3. Add StatusChanged event published when status actually changes
4. Simplify BlockStatusDot to always use BlockStatusService
5. Remove DataStateService from status display (keep for internal tracking if needed)

**Key Principle:** One object updated, all status dots point to it and update automatically.

---

## Council Analysis

### Architect Analysis

**Problem Understanding:**
The current split between BlockStatusService and DataStateService creates architectural inconsistency. Status display logic is scattered across multiple services and UI components, violating single responsibility principle.

**Key Concerns:**

1. **Service Split Violates Single Responsibility**
   - Two services doing similar things (determining block status)
   - UI components must know which service to use
   - Creates coupling between UI and service selection logic

2. **Inconsistent Abstraction Level**
   - ShowManager blocks use BlockStatus (processor-defined levels)
   - Other blocks use DataState (service-calculated freshness)
   - Same concept (block status) handled differently

3. **Event-Driven Architecture Opportunity**
   - Current system relies on cache invalidation + polling
   - StatusChanged event would be cleaner, more reactive
   - Aligns with event-driven patterns already in use

4. **Block Ownership of Status**
   - Blocks should define their own status (already in design via get_status_levels)
   - DataStateService calculates status externally - blocks don't own it
   - Moving data state checks into block status levels gives blocks ownership

**Alternatives Considered:**

1. **Keep Both Services, Improve Sync**
   - Add synchronization layer between services
   - **Rejected:** Adds complexity, doesn't solve root cause

2. **Make DataStateService Primary**
   - Convert BlockStatusService to use DataStateService
   - **Rejected:** Blocks lose ownership of status definition

3. **StatusChanged Event Only (No Service Unification)**
   - Keep both services but add StatusChanged event
   - **Rejected:** Doesn't solve the split architecture problem

**Vote: Approve with Conditions**

**Reasoning:**
The proposal correctly identifies the architectural inconsistency and proposes a clean solution. Unifying to BlockStatusService with block-owned status levels is the right abstraction. The StatusChanged event pattern aligns with existing event-driven architecture.

**Conditions:**
1. **Preserve DataStateService for Internal Use**: DataStateService may still be needed for execution logic (determining if data is stale for processing). Keep it but remove from status display.
2. **Migration Strategy**: Ensure incremental migration path - don't break existing blocks during transition
3. **Backward Compatibility**: Blocks without get_status_levels() should have sensible defaults, not errors

**Structural Benefits:**
- Single source of truth eliminates service selection logic
- Blocks own their status (proper encapsulation)
- Event-driven updates (reactive, not polling)
- Clear separation: status display vs. data freshness tracking

---

### Systems Analysis

**Problem Understanding:**
Status synchronization issues suggest timing problems, cache invalidation race conditions, or event delivery failures. The split services may have different caching strategies causing inconsistencies.

**Key Concerns:**

1. **Event Delivery Reliability**
   - StatusChanged events must be delivered reliably
   - What happens if event is missed? (UI shows stale status)
   - Need fallback mechanism or event replay capability

2. **Performance Impact**
   - StatusChanged event fires on every status change
   - Multiple status dots per block (panel, node, etc.) = multiple subscribers
   - Need to ensure event system can handle load

3. **Cache Invalidation Timing**
   - Current cache invalidation may happen too early/late
   - StatusChanged event should fire AFTER status is calculated and cached
   - Need to ensure atomic: calculate → cache → publish

4. **Status Calculation Performance**
   - get_status_levels() may be called frequently
   - Status conditions may be expensive (file system checks, etc.)
   - Caching is critical - ensure it works correctly

5. **Migration Risk**
   - Changing status system while in use could cause temporary inconsistencies
   - Need graceful degradation during migration

**Alternatives Considered:**

1. **Polling Instead of Events**
   - Status dots poll BlockStatusService periodically
   - **Rejected:** Inefficient, adds latency, doesn't solve sync issues

2. **StatusChanged with Retry Logic**
   - Add retry mechanism for missed events
   - **Rejected:** Adds complexity, events should be reliable

3. **Hybrid: Events + Periodic Refresh**
   - Events for immediate updates, periodic refresh as backup
   - **Consider:** Could be fallback during migration

**Vote: Approve with Conditions**

**Reasoning:**
The unified system will be more reliable than the current split approach. StatusChanged events are the right pattern, but need safeguards.

**Conditions:**
1. **Event Reliability**: Ensure event bus guarantees delivery or provide fallback (periodic refresh as backup during transition)
2. **Performance Monitoring**: Add logging/metrics for status calculation frequency and event delivery
3. **Graceful Degradation**: If status calculation fails, show "Unknown" not error - don't break UI
4. **Atomic Status Updates**: Ensure status calculation → cache → event publish is atomic (no race conditions)

**Stability Benefits:**
- Single cache reduces invalidation complexity
- Event-driven updates eliminate polling overhead
- Clear failure modes (status calculation fails → show default)

---

### UX Analysis

**Problem Understanding:**
Users see inconsistent status indicators - panel shows one status, node shows another. This creates confusion and reduces trust in the system.

**Key Concerns:**

1. **Status Accuracy**
   - Users need to trust status indicators
   - Inconsistent status = user confusion
   - Status must reflect actual block state

2. **Update Visibility**
   - Status changes should be immediately visible
   - Current delays (100ms, 150ms timers) create perception of lag
   - StatusChanged events should update instantly

3. **Status Meaning**
   - Users need to understand what status means
   - Block-defined status levels give blocks control over messaging
   - DataState (FRESH/STALE/NO_DATA) may not be user-friendly

4. **Error Communication**
   - Status should communicate problems clearly
   - Block-defined status can include helpful messages
   - Better than generic "STALE" or "NO_DATA"

5. **Consistency Across UI**
   - Same block should show same status everywhere
   - Panel, node, properties panel - all must match
   - Single source ensures this

**Alternatives Considered:**

1. **Keep DataState but Improve Sync**
   - Fix sync issues without changing status system
   - **Rejected:** DataState terminology (FRESH/STALE) is technical, not user-friendly

2. **Add Status Tooltips**
   - Keep current system but add better tooltips
   - **Rejected:** Doesn't solve inconsistency problem

**Vote: Approve**

**Reasoning:**
Unified system will provide consistent, accurate status everywhere. Block-defined status levels allow blocks to communicate status in user-friendly terms. StatusChanged events ensure immediate updates.

**UX Benefits:**
- Consistent status across all UI components
- User-friendly status messages (block-defined)
- Immediate updates (no perceived lag)
- Clear error communication

**No Conditions** - UX perspective fully supports the proposal.

---

### Pragmatic Analysis

**Problem Understanding:**
Current system works but has sync issues. Migration to unified system requires:
- All blocks implement get_status_levels()
- Update BlockStatusService
- Update BlockStatusDot
- Update BlockItem (node rendering)
- Migrate existing blocks incrementally

**Key Concerns:**

1. **Migration Scope**
   - How many blocks need get_status_levels() implementation?
   - Some blocks already have it (ShowManager, PyTorchAudioClassify, etc.)
   - Others need new implementation
   - Estimate: 10-15 blocks need work

2. **DataState Logic Migration**
   - DataStateService logic must move into block status conditions
   - Need to ensure logic is preserved correctly
   - Risk: Missing edge cases during migration

3. **Testing Complexity**
   - Need to test status calculation for all blocks
   - Need to test StatusChanged event delivery
   - Need to test UI synchronization
   - Significant test coverage needed

4. **Backward Compatibility**
   - What about blocks without get_status_levels()?
   - Need sensible defaults
   - Can't break existing functionality

5. **Incremental Migration**
   - Can we migrate block-by-block?
   - Or must it be all-at-once?
   - Incremental is safer but longer

**Alternatives Considered:**

1. **Big Bang Migration**
   - Migrate all blocks at once
   - **Rejected:** Too risky, hard to test

2. **Keep DataStateService, Fix Sync Only**
   - Just fix the sync issues
   - **Rejected:** Doesn't solve architectural problems, technical debt remains

3. **Hybrid During Migration**
   - Support both systems during transition
   - Blocks with get_status_levels() use BlockStatusService
   - Others use DataStateService
   - **Consider:** Could work as transition strategy

**Vote: Approve with Conditions**

**Reasoning:**
The proposal is sound but migration needs careful planning. Incremental approach is essential.

**Conditions:**
1. **Incremental Migration**: Migrate blocks in phases (3-5 blocks at a time), test thoroughly between phases
2. **Default Status for Missing Blocks**: Blocks without get_status_levels() should get sensible default (e.g., "Ready" with no conditions, or fallback to DataState during transition)
3. **Hybrid Support During Migration**: Support both systems during transition - blocks with get_status_levels() use new system, others use DataStateService temporarily
4. **Test Coverage**: Add comprehensive tests for status calculation and event delivery before full migration
5. **Rollback Plan**: Have plan to rollback if issues discovered during migration

**Implementation Complexity:**
- Medium complexity - significant but manageable
- Estimated effort: 2-3 weeks for full migration
- Risk: Medium (mitigated by incremental approach)

**Pragmatic Benefits:**
- Cleaner codebase long-term
- Easier to maintain (one system, not two)
- Better testability (clear status calculation logic)

---

## Unanimous Recommendation

**RECOMMENDATION: Proceed with Modifications**

The Council unanimously supports the goal of a unified, reliable block status system but requires specific modifications to ensure safe, incremental implementation.

### Required Modifications

1. **Hybrid Support During Migration**
   - BlockStatusService becomes primary for all blocks
   - Blocks with `get_status_levels()` use BlockStatusService
   - Blocks without `get_status_levels()` fallback to DataStateService temporarily
   - This allows incremental migration without breaking existing blocks

2. **StatusChanged Event with Fallback**
   - Add StatusChanged event for immediate updates
   - Keep periodic refresh (every 2-3 seconds) as fallback during transition
   - Remove periodic refresh once event reliability is proven

3. **Default Status for Missing Implementations**
   - Blocks without `get_status_levels()` get default "Ready" status
   - Or can temporarily use DataStateService during migration
   - No errors or missing status indicators

4. **Preserve DataStateService for Internal Use**
   - Keep DataStateService for execution logic (determining if data is stale for processing)
   - Remove from status display only
   - Status levels can call DataStateService internally if needed

5. **Incremental Migration Plan**
   - Phase 1: Add StatusChanged event, enhance BlockStatusService
   - Phase 2: Migrate 3-5 key blocks (LoadAudio, DetectOnsets, Separator)
   - Phase 3: Update BlockStatusDot and BlockItem to use unified system
   - Phase 4: Migrate remaining blocks incrementally (3-5 at a time)
   - Phase 5: Remove DataStateService from status display, remove fallback logic

### Modified Approach

**Single Source of Truth with Gradual Migration:**

1. **BlockStatusService Enhancement**
   - Add status change detection (compare old vs new status)
   - Publish StatusChanged event when status changes
   - Support fallback to DataStateService for blocks without get_status_levels()
   - Ensure all blocks can get status (not just ShowManager)

2. **StatusChanged Event**
   - New event: `StatusChanged(block_id, status: BlockStatus)`
   - Published by BlockStatusService when status actually changes
   - Includes complete status object
   - All status dots subscribe to this event

3. **Block Migration**
   - Each block implements `get_status_levels()` with:
     - Configuration checks (errors, warnings)
     - Data state checks (using DataStateService internally if needed)
     - Clear priority ordering
   - Migrated blocks use BlockStatusService exclusively
   - Non-migrated blocks use DataStateService fallback

4. **Simplified Status Dots**
   - BlockStatusDot always uses BlockStatusService
   - Subscribes to StatusChanged events
   - No block type checking needed
   - Fallback refresh during migration (remove after proven)

5. **Node Rendering**
   - BlockItem uses BlockStatusService (already does for ShowManager)
   - Subscribes to StatusChanged events
   - Consistent with panel status dots

### Success Criteria

- [ ] All status dots (panel, node, properties) show same status for same block
- [ ] Status updates immediately when block state changes
- [ ] No status calculation errors break UI
- [ ] All blocks have status (migrated or fallback)
- [ ] StatusChanged events deliver reliably
- [ ] Performance acceptable (status calculation < 50ms, events < 10ms)

### Implementation Plan

**Phase 1: Foundation (Week 1)**
- Add StatusChanged event
- Enhance BlockStatusService with change detection and event publishing
- Add fallback support for blocks without get_status_levels()
- Update BlockStatusDot to subscribe to StatusChanged
- Add periodic refresh fallback (remove after proven)

**Phase 2: Key Blocks Migration (Week 1-2)**
- Migrate LoadAudio, DetectOnsets, Separator blocks
- Test status calculation and event delivery
- Verify UI synchronization

**Phase 3: UI Updates (Week 2)**
- Update BlockItem to use unified system
- Remove block type checking from status dots
- Test across all UI components

**Phase 4: Remaining Blocks (Week 2-3)**
- Migrate remaining blocks incrementally (3-5 at a time)
- Test after each batch
- Monitor for issues

**Phase 5: Cleanup (Week 3)**
- Remove DataStateService from status display
- Remove fallback logic
- Remove periodic refresh
- Final testing and documentation

### Key Strengths

- **Architect**: Clean single source of truth, proper block ownership
- **Systems**: Event-driven updates, reliable caching, graceful degradation
- **UX**: Consistent status everywhere, immediate updates, user-friendly messages
- **Pragmatic**: Incremental migration, backward compatible, testable

### Risks and Mitigations

**Risk**: Migration breaks existing blocks
- **Mitigation**: Hybrid support, fallback to DataStateService, incremental migration

**Risk**: StatusChanged events missed
- **Mitigation**: Periodic refresh fallback during transition, event reliability testing

**Risk**: Performance degradation
- **Mitigation**: Caching, status calculation optimization, performance monitoring

**Risk**: Status calculation errors break UI
- **Mitigation**: Try-catch with default status, graceful degradation

---

## Next Steps

1. Review and approve this council decision
2. Create detailed implementation plan with specific tasks
3. Begin Phase 1: Foundation work
4. Test incrementally after each phase
5. Document status level implementation pattern for block developers

---

**Council Members:**
- Architect: Approve with Conditions ✓
- Systems: Approve with Conditions ✓
- UX: Approve ✓
- Pragmatic: Approve with Conditions ✓

**Unanimous Recommendation: Proceed with Modifications**
