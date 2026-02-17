# Feature Development Guide

## Step-by-Step Workflow

### Phase 1: Planning

**1. Define the Problem**
- What specific user problem does this solve?
- Do we have evidence (feedback, usage data)?
- What happens if we don't build this?

**2. Identify MVP**
- What's the minimum that would work?
- What's the 20% that gives 80% value?
- What can wait for later iterations?

**3. Consider Alternatives**
- Can we solve this with existing features?
- Can we remove something instead?
- Is there a simpler approach?

**4. Evaluate Scope**
- Estimate lines of code
- Identify new dependencies
- Assess testing complexity
- Consider maintenance burden

**5. Council Review (if major)**
- Use `modules/process/council/` framework
- Get approval before major implementation
- Minor features can proceed without council

### Phase 2: Design

**1. Architecture Review**
- Where does this fit in layered architecture?
- What layers are affected?
- Are boundaries respected?

**2. Component Design**
- What components are needed?
- How do they interact?
- What's the data flow?

**3. Interface Design**
- How do users interact with this?
- Is it consistent with existing patterns?
- Are error messages clear?

### Phase 3: Implementation

**1. Start with Domain**
- Create entities/value objects if needed
- Define repository interfaces
- Keep domain pure (no infrastructure)

**2. Implement Application Layer**
- Create services/processors
- Implement business logic
- Add command handlers if needed

**3. Add Infrastructure**
- Implement repositories
- Add persistence if needed
- Handle external dependencies

**4. Create UI (if needed)**
- Follow existing UI patterns
- Use settings_abstraction if configurable
- Ensure cleanup of resources

**5. Write Tests**
- Unit tests for logic
- Integration tests for workflows
- Test error cases

### Phase 4: Integration

**1. Register Components**
- Register blocks in BlockRegistry
- Register commands if needed
- Register UI panels if needed

**2. Update Documentation**
- Update docs/ if architecture changed
- Update `core/CURRENT_STATE.md` if capabilities changed
- Add usage examples

**3. Test End-to-End**
- Test in CLI
- Test in GUI (if applicable)
- Test error scenarios
- Verify resource cleanup

### Phase 5: Validation

**1. Code Review**
- Self-review against core values
- Check for unnecessary complexity
- Verify cleanup and error handling

**2. User Testing**
- Does it solve the problem?
- Is it intuitive?
- Are errors clear?

**3. Performance Check**
- No memory leaks
- Reasonable resource usage
- Acceptable performance

## Common Feature Types

### Adding a New Block

1. Create processor in `src/application/blocks/`
2. Register in `BlockRegistry`
3. Create UI panel in `ui/qt_gui/block_panels/`
4. Add settings if needed
5. Implement cleanup()
6. Add tests

See: `modules/patterns/block_implementation/`

### Adding Settings

1. Define settings schema (dataclass)
2. Create settings manager
3. Integrate in UI components
4. Add validation

See: `modules/patterns/settings_abstraction/`

### Adding a Command

1. Create command class (QUndoCommand)
2. Register in CommandBus
3. Add to ApplicationFacade
4. Add CLI support
5. Add tests

### Adding UI Component

1. Follow existing UI patterns
2. Use settings_abstraction if configurable
3. Ensure proper cleanup
4. Test in both light/dark themes

See: `modules/patterns/ui_components/`

## Iteration Strategy

**Start Small:**
- Build MVP first
- Get it working
- Test with users

**Iterate Based on Usage:**
- What do users actually use?
- What's missing?
- What's unnecessary?

**Refine Before Expanding:**
- Polish MVP before adding features
- Remove unused parts
- Simplify based on feedback

## Remember

- **Question necessity** - Does this solve a real problem?
- **Start minimal** - MVP first, iterate later
- **Incremental delivery** - Small, testable steps
- **Consider deletion** - Can we remove instead?
- **Align with values** - Simplicity over complexity

