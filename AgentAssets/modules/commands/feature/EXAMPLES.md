# Feature Development Examples

## Example 1: Adding Settings to Existing Block

**Feature:** Add model selection to SeparatorBlock

**Problem:** Users want to choose different Demucs models for quality/speed trade-offs.

**MVP:** Add model setting with 3 most common models.

**Implementation:**
1. Define `SeparatorSettings` dataclass with `model` field
2. Create `SeparatorSettingsManager` with validation
3. Add model dropdown to SeparatorPanel
4. Update SeparatorBlockProcessor to use setting

**Result:** Simple addition, follows settings_abstraction pattern, minimal complexity.

---

## Example 2: New Block Type

**Feature:** Add AudioNormalize block

**Problem:** Users need to normalize audio levels.

**MVP:** Simple peak normalization with target level setting.

**Implementation:**
1. Create `NormalizeBlockProcessor` in `src/application/blocks/`
2. Register in `BlockRegistry` with audio input/output
3. Create `NormalizePanel` UI
4. Add settings for target level
5. Implement cleanup()
6. Add tests

**Result:** Follows block_implementation pattern, clean separation of concerns.

---

## Example 3: Feature That Required Refactoring

**Feature:** Add undo/redo support

**Problem:** Users need to undo operations.

**MVP:** Undo/redo for block add/delete/connect operations.

**Implementation:**
1. Refactor commands to use QUndoCommand (required refactoring)
2. Create CommandBus for command execution
3. Add undo/redo to ApplicationFacade
4. Add UI controls for undo/redo

**Result:** Required council review due to architectural change, but enabled many future features.

---

## Lessons Learned

**From Example 1:**
- Using existing patterns (settings_abstraction) made implementation straightforward
- MVP scope kept it simple
- Easy to extend later with more models

**From Example 2:**
- Following block_implementation pattern ensured consistency
- Cleanup() prevented resource leaks
- Tests caught issues early

**From Example 3:**
- Major architectural changes need council review
- Refactoring enabled future features (good investment)
- Incremental rollout (started with basic commands)

## Anti-Examples (What Not to Do)

### Over-Engineering

**Bad:** Creating a generic "ProcessingBlock" abstraction before we have 3+ block types that need it.

**Good:** Start with concrete blocks, abstract when pattern emerges 3+ times.

### Scope Creep

**Bad:** "While we're adding settings, let's also add validation, UI hints, and advanced options."

**Good:** Add settings first, validate it works, then consider enhancements.

### Premature Optimization

**Bad:** "Let's add caching, async processing, and batch operations from the start."

**Good:** Build simple version first, optimize only if performance becomes an issue.


