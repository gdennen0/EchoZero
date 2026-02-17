# Editor Refactoring Proposal - Council Decision

## Proposal

Refactor Editor block and TimelineWidget to implement individual layer/event management instead of bulk clearing/recreating on execution. Clarify, simplify, and standardize all Editor/TimelineWidget processes with clear expected inputs and explicit errors. Consider merging EditorPanel and TimelineWidget for a cleaner architecture.

## Problem Statement

### Current Issues

1. **Bulk Clear/Recreate Anti-Pattern**: Execution clears all layers then recreates them, losing state and causing sync issues
2. **Unclear Boundaries**: EditorPanel and TimelineWidget are separate, causing confusion about ownership and responsibilities
3. **Inconsistent Error Handling**: Missing explicit error handling for invalid inputs
4. **Multiple Entry Points**: Layer/event operations happen from many places without clear single API
5. **State Preservation Problems**: Synced layers and non-synced layers handled inconsistently during execution

### Evidence

- User feedback: "layers still were saved/maintained throughout the execution" when filters unselected
- Code complexity: Layer creation happens from 3+ different places (EditorCreateLayerCommand, _restore_layer_state, TimelineWidget.set_events)
- Execution issues: Layers persist when they should be cleared based on filters
- Architecture confusion: TimelineWidget and EditorPanel have overlapping responsibilities

## Proposed Solution

### Core Changes

1. **Individual Layer Management**: Replace bulk clear with incremental add/remove/update operations
2. **Unified API**: Single set of clearly defined functions for all layer/event operations
3. **Clear Input Validation**: Explicit error throwing for invalid states
4. **Architecture Simplification**: Consider merging EditorPanel + TimelineWidget (or clearly separate concerns)
5. **Standardized Operations**: Consistent patterns for all layer/event management

### Proposed API Structure

#### Layer Functions (Single Entry Points)
- `add_layer(name, properties)` - Create new layer
- `remove_layer(layer_id)` - Delete layer and all its events
- `update_layer(layer_id, properties)` - Change layer settings (name, color, height, etc.)
- `move_layer(layer_id, new_index)` - Change layer position/order
- `move_layer_to_eventdataitem(layer_id, target_eventdataitem_id)` - Move layer to different EventDataItem

#### Event Functions (Single Entry Points)
- `add_event(time, duration, classification, layer_id, metadata)` - Add single event
- `remove_event(event_id)` - Delete single event
- `update_event(event_id, properties)` - Change event properties
- `move_event_to_layer(event_id, target_layer_id)` - Move event between layers (EXISTS - preserve this pattern)

### Execution Flow Changes

**Current**: Clear all → Pull data → Recreate all
**Proposed**: Compare state → Incrementally add/remove/update layers and events

## Council Analysis

### Architect Analysis

**Structural Concerns:**
- Current EditorPanel/TimelineWidget separation creates unclear boundaries - responsibilities overlap
- Bulk operations violate single responsibility - should be incremental
- Multiple entry points for same operations create maintenance burden
- Lack of clear API contracts makes debugging difficult

**Architectural Benefits:**
- Unified API provides clear boundaries and responsibilities
- Incremental operations enable better state preservation
- Explicit errors improve debuggability
- Merging or clear separation would reduce coupling

**Key Questions:**
1. Should EditorPanel and TimelineWidget merge? **Analysis needed**
2. Where should command execution live - EditorPanel or unified component?
3. How to handle synced layers during incremental updates?
4. What's the migration path from bulk to incremental?

**Alternatives Considered:**
1. **Keep separate but clarify boundaries** - Easier migration, but doesn't solve coupling
2. **Merge completely** - Cleanest, but large refactor
3. **Incremental without merging** - Balanced approach, clarify APIs first

**Vote:** ✅ **Approve with Conditions**

**Conditions:**
- Phase 1: Implement unified API while keeping separate components (validate approach)
- Phase 2: Evaluate merge vs. separation based on Phase 1 learnings
- Preserve existing event movement logic (it works well - use as template)
- Define clear API contracts with explicit error types
- Document migration from existing code paths

---

### Systems Engineer Analysis

**Infrastructure Concerns:**
- Bulk clear/recreate operations are inefficient (unnecessary work)
- State preservation issues during execution (synced layers lost)
- No clear error recovery paths when operations fail partially
- Performance: Incremental updates should be faster than full recreate

**Failure Modes:**
- Partial update failures (some layers updated, others failed)
- State inconsistency during incremental updates
- Race conditions if updates happen from multiple sources
- Error recovery: How to rollback partial updates?

**Performance:**
- Incremental operations should improve performance (only update what changed)
- Need batching strategy for multiple updates
- Memory: Better state tracking vs. current approach

**Key Questions:**
1. How to ensure atomic operations for incremental updates?
2. Error recovery strategy for partial failures?
3. Performance testing needed for incremental vs. bulk?

**Vote:** ✅ **Approve with Conditions**

**Conditions:**
- Implement transaction/batching system for multiple incremental updates
- Comprehensive error handling with rollback capability
- Performance benchmarks: incremental vs. current approach
- Clear error recovery paths documented
- Protect synced layers during all operations (execution, filters, etc.)

---

### UX Engineer Analysis

**User Experience Concerns:**
- Current behavior is confusing: "layers still saved" when filter cleared
- No clear feedback when operations fail
- Complex mental model (when do layers persist? when do they clear?)

**Benefits:**
- Predictable behavior: operations are explicit and clear
- Better error messages (explicit errors instead of silent failures)
- Consistent operations: same API for all layer/event management

**Key Questions:**
1. How to communicate incremental updates to users?
2. Error message clarity for invalid operations?
3. Undo/redo behavior with incremental operations?

**Vote:** ✅ **Approve**

**Reasoning:**
- Explicit errors improve UX (users know what went wrong)
- Predictable incremental operations are easier to understand than bulk recreate
- Clear API enables better error messages

**Suggestions:**
- Ensure all operations provide clear user feedback
- Error messages should guide users to fix issues
- Undo/redo should work seamlessly with incremental operations

---

### Pragmatic Engineer Analysis

**Implementation Complexity:**
- Large refactor but necessary for maintainability
- Can be done incrementally (start with unified API, then merge later)
- Existing code has patterns to learn from (event movement works well)

**Testing:**
- Need comprehensive tests for each operation
- Test synced layer preservation during all scenarios
- Test error cases (invalid inputs, missing layers, etc.)

**Scope:**
- MVP: Unified API + incremental execution (Phase 1)
- Full: Merge components if Phase 1 validates (Phase 2)

**Risk Assessment:**
- **Medium Risk**: Large refactor but can be done incrementally
- **Migration Risk**: Need to migrate existing code paths gradually
- **Testing Risk**: Complex scenarios (synced layers, filters, execution)

**Key Questions:**
1. Can we preserve existing working code (event movement) as template?
2. Incremental migration strategy?
3. Testing approach for complex scenarios?

**Vote:** ✅ **Approve with Conditions**

**Conditions:**
- Phase 1: Implement unified API, keep components separate
- Learn from existing event movement code (it works - use as pattern)
- Incremental migration: One operation at a time
- Comprehensive test coverage before merging components
- Document all existing code paths before refactoring (learn from them)

---

## Council Recommendation

**RECOMMENDATION: Proceed with Phased Approach**

The Council unanimously approves this refactoring with a phased implementation strategy.

### Key Strengths

- **Architect**: Unified API clarifies boundaries and responsibilities
- **Systems**: Incremental operations improve efficiency and state management
- **UX**: Explicit errors and predictable operations improve user experience
- **Pragmatic**: Can be done incrementally with learnings from existing code

### Implementation Plan

#### Phase 1: Unified API (Weeks 1-2)

**Goal**: Establish clear API contracts while keeping existing architecture

1. **Document Current Patterns**
   - Analyze existing event movement code (it works well - use as template)
   - Document all current layer/event operation entry points
   - Identify which patterns work and which don't

2. **Define Unified API**
   - Create clear function signatures for all operations
   - Define explicit error types for invalid inputs
   - Document expected inputs and validation rules

3. **Implement Layer Functions**
   - `add_layer()` - Single entry point for layer creation
   - `remove_layer()` - Single entry point for layer deletion
   - `update_layer()` - Single entry point for layer property changes
   - `move_layer()` - Single entry point for layer reordering

4. **Implement Event Functions**
   - `add_event()` - Single entry point for event creation
   - `remove_event()` - Single entry point for event deletion
   - `update_event()` - Single entry point for event property changes
   - Preserve existing `move_event_to_layer()` (works well)

5. **Migrate Execution to Incremental**
   - Replace bulk clear with compare-and-update logic
   - Preserve synced layers during all operations
   - Add explicit error handling for invalid states

#### Phase 2: Architecture Evaluation (Weeks 3-4)

**Goal**: Evaluate whether to merge EditorPanel + TimelineWidget

1. **Evaluate Phase 1 Results**
   - Does unified API solve the boundary issues?
   - Are there still coupling problems?
   - User feedback on predictability

2. **Decision Point**
   - If unified API is sufficient → Keep separate, clarify boundaries
   - If coupling remains problematic → Plan merge

3. **If Merging:**
   - Design unified component architecture
   - Migration plan from two components to one
   - Preserve all functionality during merge

### Required Conditions

- [ ] **Learn from Existing Code**: Document how event movement works (it's correct) before refactoring
- [ ] **Preserve Synced Layers**: All operations must protect synced layers
- [ ] **Explicit Errors**: All invalid inputs throw clear, actionable errors
- [ ] **Incremental Migration**: Migrate one operation at a time, test thoroughly
- [ ] **Comprehensive Testing**: Test all scenarios (synced layers, filters, execution)
- [ ] **Performance Validation**: Benchmark incremental vs. bulk operations
- [ ] **Error Recovery**: Define rollback strategy for partial failures

### Success Criteria

1. **Predictable Behavior**: Layers/events persist or clear based on explicit operations, not implicit execution
2. **Clear Errors**: All invalid states throw explicit errors with actionable messages
3. **Synced Layer Protection**: Synced layers persist through all operations (execution, filters, etc.)
4. **Unified API**: All layer/event operations go through single, well-defined functions
5. **Maintainable Code**: Future developers can easily find and understand operations

### Documentation Requirements

1. **API Documentation**: Clear function signatures with input validation rules
2. **Architecture Decision Record**: Why incremental vs. bulk, merge vs. separate
3. **Migration Guide**: How existing code paths map to new API
4. **Error Reference**: All error types and recovery strategies
5. **Testing Strategy**: How to test incremental operations and edge cases

### Risks and Mitigations

**Risk 1: Breaking Existing Functionality**
- **Mitigation**: Incremental migration, comprehensive testing, preserve event movement pattern

**Risk 2: Performance Regression**
- **Mitigation**: Benchmark before/after, optimize hot paths, batch operations

**Risk 3: Synced Layer Issues**
- **Mitigation**: Explicit synced layer protection in all operations, comprehensive testing

**Risk 4: Migration Complexity**
- **Mitigation**: Document all existing code paths first, migrate incrementally

---

## Next Steps

1. **Document Existing Patterns** (This week)
   - Analyze event movement code (reference implementation)
   - Document all layer/event operation entry points
   - Create mapping of current → proposed API

2. **Design Unified API** (This week)
   - Define function signatures
   - Define error types
   - Create API documentation template

3. **Implement Phase 1** (Weeks 1-2)
   - Start with layer operations
   - Then event operations
   - Finally execution incremental logic

4. **Evaluate and Decide Phase 2** (Week 3)
   - Assess Phase 1 results
   - Make merge vs. separate decision

---

**Council Status**: ✅ **Unanimously Approved with Phased Implementation**

**Documented**: December 2025
**Status**: Ready for Implementation
