# Research Investigation Guide

## Step-by-Step Research Workflow

### Phase 1: Problem Exploration

**1. Start with Questions, Not Solutions**

Before researching solutions, understand the problem:
- What problem are we trying to understand?
- Why is this problem worth solving?
- What evidence do we have that this is a real problem?
- What happens if we don't solve it?

**2. Identify and Question Assumptions**

List all assumptions you're making:
- What do we assume is true?
- What do we assume users need?
- What do we assume is technically possible?
- Can we validate these assumptions?

**3. Explore the Current State**

Understand what exists:
- What already exists in EchoZero?
- What patterns are already established?
- What's missing or broken?
- What works well?

**4. Research Similar Systems**

Learn from others:
- How do similar systems solve this?
- What patterns exist in the domain?
- What can we learn from existing solutions?
- What mistakes should we avoid?

### Phase 2: Technical Investigation

**1. Architecture Analysis**

Understand architectural implications:
- Where would this fit in the layered architecture?
- What layers would be affected?
- Are there architectural constraints?
- What boundaries need to be respected?

**2. Feasibility Research**

Assess technical feasibility:
- Is this technically possible?
- What are the technical challenges?
- What dependencies would be needed?
- What are the performance implications?
- What's the maintenance burden?

**3. Pattern Research**

Investigate existing patterns:
- Are there patterns in EchoZero we can reuse?
- What patterns exist in similar domains?
- What anti-patterns should we avoid?
- How have we solved similar problems?

**4. Codebase Exploration**

Search the codebase for relevant code:
- Are there similar implementations?
- What patterns are used?
- What interfaces exist?
- What abstractions are available?

### Phase 3: Alternative Exploration

**1. Generate Multiple Alternatives**

Don't settle on the first idea:
- What are different ways to solve this?
- What's the simplest approach?
- Can we solve this without building anything?
- Can we remove something instead?
- Can we extend existing features?

**2. Compare Approaches**

Evaluate each alternative:
- What are the trade-offs?
- What's the complexity?
- What's the maintenance burden?
- What's the user experience?
- What's the implementation effort?

**3. Evaluate Against Core Values**

Apply EchoZero principles:
- Which approach embodies "best part is no part"?
- Which is simplest?
- Which reduces complexity?
- Which can be deleted later if not needed?
- Which aligns with existing patterns?

**4. Consider Deletion First**

Always ask:
- Can we solve this by removing something?
- Can we simplify instead of adding?
- Can we reduce complexity elsewhere?

### Phase 4: Validation and Documentation

**1. Validate Assumptions**

Test your assumptions:
- What did research confirm?
- What did research contradict?
- What new questions emerged?
- What risks were identified?

**2. Gather Evidence**

Collect supporting information:
- What evidence supports proceeding?
- What evidence suggests not proceeding?
- What information is still missing?
- What would we need to know to decide?

**3. Document Findings**

Create research document:
- Use the PRESET.md template
- Document key insights
- Record alternatives explored
- Note validated/contradicted assumptions
- List open questions
- Provide recommendations

**4. Determine Next Steps**

Decide on action:
- Proceed to feature development?
- Need more research?
- Abandon idea?
- Explore different direction?

## Research Methodologies

### Codebase Search Strategy

1. **Semantic Search**
   - Search for concepts, not just keywords
   - Look for similar functionality
   - Find related patterns

2. **Pattern Discovery**
   - Identify common patterns
   - Find abstraction opportunities
   - Discover reusable components

3. **Interface Exploration**
   - Find existing interfaces
   - Understand contracts
   - Identify extension points

### External Research Strategy

1. **Domain Research**
   - Research similar systems
   - Study domain patterns
   - Learn from industry practices

2. **Technical Research**
   - Investigate libraries/frameworks
   - Research performance implications
   - Study architectural patterns

3. **User Research**
   - Gather user feedback
   - Understand pain points
   - Validate assumptions

### Validation Strategy

1. **Assumption Testing**
   - List assumptions explicitly
   - Find evidence for/against each
   - Identify unvalidated assumptions

2. **Proof of Concept**
   - Build small prototypes if needed
   - Test technical feasibility
   - Validate approach

3. **Expert Consultation**
   - Discuss with team
   - Use council process if major
   - Get feedback on approach

## Common Research Scenarios

### Researching a New Block Idea

1. **Explore Existing Blocks**
   - How are blocks structured?
   - What patterns do they follow?
   - What interfaces exist?

2. **Investigate Technical Requirements**
   - What processing is needed?
   - What dependencies are required?
   - What are performance implications?

3. **Consider Alternatives**
   - Can existing blocks be extended?
   - Can multiple blocks be combined?
   - Is a new block necessary?

4. **Validate Need**
   - Is there user demand?
   - Does it solve a real problem?
   - Is it aligned with core values?

### Researching a UI Feature

1. **Explore Existing UI Patterns**
   - How are similar features implemented?
   - What UI patterns exist?
   - What components are available?

2. **Investigate User Experience**
   - How do users currently work?
   - What's the workflow?
   - What would improve it?

3. **Consider Alternatives**
   - Can existing UI be improved?
   - Can we simplify instead?
   - Is new UI necessary?

### Researching a Refactoring Opportunity

1. **Identify the Problem**
   - What's the concrete problem?
   - What evidence supports refactoring?
   - What's the pain point?

2. **Explore Current Structure**
   - How is it currently organized?
   - What patterns exist?
   - What's working well?

3. **Investigate Alternatives**
   - Can we delete instead?
   - Can we simplify?
   - What's the simplest fix?

## Transitioning to Feature Development

When research is complete and you've decided to proceed:

### 1. Create Research Summary

Distill key findings into actionable insights:
- What did we learn?
- What assumptions were validated/contradicted?
- Which alternative is recommended and why?
- What risks or constraints were identified?

### 2. Reference Research in Feature Proposal

Link research document and summarize relevant findings:

```markdown
Feature Proposal: [Name]

Research Completed: See [link to research document]

Key Research Findings:
- [Finding 1 that informed this proposal]
- [Finding 2 that informed this proposal]
- [Finding 3 that informed this proposal]

Validated Assumptions:
- [Assumption 1] ✓ Confirmed
- [Assumption 2] ✗ Contradicted (adjusted approach)
- [Assumption 3] ⚠ Partially validated (monitoring)

Recommended Approach: [Based on research Alternative X]

Reason for Selection:
[Why this alternative from research]

[Continue with standard feature template...]
```

### 3. Use Research to Define Scope

Apply research findings to feature planning:
- **MVP Definition:** Research shows what's truly necessary
- **Alternative Selection:** Use research comparison to choose approach
- **Risk Mitigation:** Address risks identified in research
- **Pattern Application:** Use patterns discovered during research

### 4. Validate Assumptions During Development

Keep research document handy during implementation:
- Test assumptions as you build
- Verify technical feasibility claims
- Confirm performance expectations
- Watch for risks identified in research

### 5. Update Research if Needed

If implementation reveals new information:
- Update research document with learnings
- Document what was correct/incorrect
- Help future research be more accurate

### Example Transition

**Research Phase:**
```
Research: How should we implement undo/redo for timeline edits?

Findings:
- Command pattern is established in codebase
- Qt provides QUndoCommand base class
- Three approaches investigated
- Memento pattern would require significant refactor
- Command pattern fits existing architecture

Recommendation: Use QUndoCommand pattern
```

**Feature Phase:**
```
Feature: Timeline Undo/Redo

Research: See research/timeline-undo-redo.md

Key Finding: Command pattern already used in codebase,
QUndoCommand provides undo/redo infrastructure

Approach: Extend existing QUndoCommand pattern

MVP:
- Undo/redo for move event
- Undo/redo for create/delete event
- Standard Ctrl+Z / Ctrl+Y shortcuts

[Continue with feature planning...]
```

## Remember

- **Question first, solve later** - Understand the problem before researching solutions
- **Explore alternatives** - Don't settle on the first idea
- **Validate assumptions** - Test what you think you know
- **Consider deletion** - Always ask if we can remove instead
- **Document findings** - Research is useless if not documented
- **Use research to inform** - Let findings guide feature development

Research helps avoid building unnecessary features by thoroughly investigating ideas before committing to implementation.
