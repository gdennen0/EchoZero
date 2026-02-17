# Feature Development Preset

## Feature Proposal Template

**Feature Name:** [Clear, descriptive name]

**Problem:** [What user problem does this solve? Be specific.]

**Evidence:** [User feedback, usage data, clear pain point]

**Proposed Solution:**
- [Component 1]
- [Component 2]
- [Component 3]

**Alternatives Considered:**
1. [Alternative 1] - [Why not chosen]
2. [Alternative 2] - [Why not chosen]

**Scope:**
- **MVP:** [Minimum viable version - what's the 20% that gives 80% value?]
- **Future:** [What can wait?]

**Estimated Effort:**
- LOC: [Estimate]
- Dependencies: [List]
- Testing: [Complexity]
- Maintenance: [Burden]

**Council Review Required:** [Yes/No - Major changes need council]

---

## Feature Development Checklist

### Planning Phase
- [ ] Problem clearly defined with evidence
- [ ] MVP scope identified (20% for 80% value)
- [ ] Alternatives considered
- [ ] Council review completed (if major)
- [ ] Implementation plan created

### Implementation Phase
- [ ] Follow layered architecture (domain → application → infrastructure)
- [ ] Create tests alongside code
- [ ] Add settings if needed (use settings_abstraction module)
- [ ] Implement UI components if needed
- [ ] Add cleanup() if resources are used

### Validation Phase
- [ ] Tests pass
- [ ] Manual testing completed
- [ ] Error handling verified
- [ ] Resource cleanup verified
- [ ] Documentation updated

### Completion Phase
- [ ] Code review (self or council)
- [ ] Encyclopedia updated (if architecture changed)
- [ ] `core/CURRENT_STATE.md` updated (if capabilities changed)
- [ ] Feature works in both CLI and GUI (if applicable)

---

## Red Flags (Reconsider)

- "It would be cool if..."
- "Users might want..."
- "While we're at it..."
- No clear user problem
- MVP scope is too large
- Adds significant complexity without clear benefit

---

## Green Flags (Proceed)

- Clear user problem with evidence
- MVP is minimal and focused
- Incremental delivery possible
- Aligns with core values
- Reduces complexity elsewhere

