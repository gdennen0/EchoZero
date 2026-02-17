# Council Proposal: Add Project-Level Actions to Action Sets

## Proposal

**Title:** Refactor ActionItem to Support Both Block and Project-Level Actions

**Problem:**
Currently, ActionItem requires a `block_id`, making it impossible to include project-level actions (like `execute_project`, `save_project`, `validate_project`) in action sets. Users want to create action sequences that include both block-specific actions and project-level operations.

**Concrete Evidence:**
- ActionItem entity has `block_id: str` as required field
- `discover_setlist_actions` only discovers block-based quick_actions
- Action set editor UI only shows blocks in dropdown
- Project-level operations exist (execute_project, save_project, validate_project) but can't be sequenced

**Proposed Solution:**
Make `block_id` optional in ActionItem and add `action_type` field to distinguish:
- `action_type: "block"` - action applies to a specific block (requires block_id)
- `action_type: "project"` - action applies to project level (block_id is None/empty)

**Implementation Plan:**
1. Update ActionItem entity: make block_id Optional[str], add action_type field
2. Update database schema: make block_id nullable, add action_type column
3. Update discovery: add project-level actions alongside block actions
4. Update UI: show "Project" as option in block dropdown, show project actions when selected
5. Update execution: route to project-level facade methods when action_type is "project"

**Estimated Effort:** ~500 LOC changes across entity, repository, service, UI layers

**New Dependencies:** None

---

## Council Analysis

### Architect Analysis

**Structural Concerns:**

**Positive:**
- Clean extension of existing pattern
- Maintains single ActionItem entity (no duplication)
- Follows existing action discovery/execution flow

**Concerns:**
1. **Type safety:** Optional block_id could lead to runtime errors if not validated
2. **Polymorphism:** ActionItem now represents two different concepts (block vs project actions)
3. **Discovery pattern:** Need to extend discovery beyond block-based quick_actions
4. **Execution routing:** Need conditional logic to route to block vs project methods

**Alternative Structural Approach:**
- Keep ActionItem block-only
- Create separate ProjectActionItem entity
- Use union type or base class for ActionSet.actions
- More explicit, but adds complexity

**Impact on Architecture:**
- As proposed: Single entity with optional field - simpler but less type-safe
- With alternative: Two entities - more explicit but more complex

**Vote: Approve with Conditions**

**Reasoning:**
The proposed approach is simpler and maintains a single action flow. However, we need strong validation to prevent runtime errors.

**Conditions:**
1. Add validation in ActionItem.__post_init__ to ensure action_type and block_id are consistent
2. Add type checking in execution layer before routing
3. Update all ActionItem creation sites to set action_type explicitly
4. Add database migration to add action_type column (new databases only)

---

### Systems Analysis

**Infrastructure Concerns:**

**Positive:**
- No new dependencies
- Database change is simple (nullable column + new column)
- Execution routing is straightforward

**Concerns:**
1. **Database migration:** Add action_type column to schema
2. **Validation:** Must prevent invalid combinations (project action with block_id, block action without block_id)
3. **Error handling:** Clear errors when action_type/block_id mismatch

**Resource Analysis:**
- Database: Minimal (one nullable column, one new column)
- Memory: No change
- Performance: No impact

**Failure Modes:**
- Invalid action_type value
- Block action without block_id
- Project action with block_id
- Missing project-level action handler

**Vote: Approve with Conditions**

**Reasoning:**
Technically straightforward but requires careful validation and migration.

**Conditions:**
1. Database migration to add action_type column
2. Validation in ActionItem to catch invalid combinations
3. Comprehensive error handling in execution layer
4. Unit tests for all validation paths

---

### UX Analysis

**User Experience Concerns:**

**Positive:**
- Enables more powerful action sequences
- Natural extension of existing UI
- Users can sequence: validate → execute → save

**Concerns:**
1. **UI clarity:** Need to clearly distinguish block vs project actions
2. **Discovery:** How do users discover project actions?
3. **Naming:** "Project" as block name might be confusing
4. **Workflow:** Does this match user mental model?

**UI Design:**
- Block dropdown: Add "-- Project Actions --" as first option
- When selected, action dropdown shows project-level actions
- Clear labeling: "Project: Execute" vs "Block: Set Model"

**Vote: Approve**

**Reasoning:**
Clear UX benefit with straightforward UI changes. The "-- Project Actions --" option makes it discoverable.

---

### Pragmatic Engineer Analysis

**Practical Concerns:**

**Positive:**
- Small, focused change
- Clear boundaries
- Can be implemented incrementally

**Concerns:**
1. **Scope:** Are we sure we need this? What's the concrete use case?
2. **Testing:** Need tests for both action types
3. **Documentation:** Update all action-related docs
4. **Migration:** Handle existing data gracefully

**Complexity Assessment:**
- Entity change: Low (make field optional, add field)
- Repository change: Low (update queries, add migration)
- Service change: Medium (extend discovery, add project actions)
- UI change: Medium (update dropdowns, add project option)
- Execution change: Medium (add routing logic)

**Testing Strategy:**
- Unit tests for ActionItem validation
- Integration tests for discovery (block + project)
- Integration tests for execution (both types)
- Database schema migration test

**Vote: Approve with Conditions**

**Reasoning:**
Reasonable complexity, clear implementation path. Need to ensure migration and testing are thorough.

**Conditions:**
1. Implement incrementally: entity → repository → service → UI → execution
2. Test each layer before moving to next
3. Migration script tested on sample data
4. Documentation updated for new action_type field

---

## Unanimous Recommendation

**Vote: Approve with Conditions**

**Summary:**
All council members approve the proposal with conditions focused on validation, migration, and testing. The approach is sound and maintains simplicity while extending functionality.

**Required Conditions:**
1. ✅ Add ActionItem validation for action_type/block_id consistency
2. ✅ Database migration to add action_type column
3. ✅ Type checking in execution layer
4. ✅ Comprehensive error handling
5. ✅ Unit and integration tests
6. ✅ Incremental implementation with testing at each layer
7. ✅ Documentation updates

**Implementation Order:**
1. Entity layer: Update ActionItem (make block_id Optional, add action_type)
2. Repository layer: Update schema, add migration
3. Service layer: Extend discovery to include project actions
4. UI layer: Add "Project" option to block dropdown
5. Execution layer: Add routing for project actions

**Risk Assessment:**
- **Low Risk:** Well-scoped change, clear boundaries, no new dependencies
- **Mitigation:** Strong validation, comprehensive testing, incremental rollout

---

## Implementation Notes

**ActionItem Changes:**
```python
@dataclass
class ActionItem:
    action_type: str  # "block" or "project"
    block_id: Optional[str] = None  # Required if action_type == "block"
    # ... rest of fields
```

**Project Actions to Include:**
- execute_project
- validate_project  
- save_project
- (Future: export_project, etc.)

**Discovery Extension:**
```python
def discover_available_actions(project_id: str):
    # Existing block actions
    block_actions = {...}
    
    # Add project actions
    project_actions = {
        "project": {
            "block_name": "Project",
            "block_type": "Project",
            "actions": [
                {"name": "execute_project", "description": "Execute entire project"},
                {"name": "validate_project", "description": "Validate project graph"},
                {"name": "save_project", "description": "Save project changes"},
            ]
        }
    }
    
    return {**block_actions, **project_actions}
```

---

*Council Decision Date: [Current Date]*
*Status: Approved with Conditions*

