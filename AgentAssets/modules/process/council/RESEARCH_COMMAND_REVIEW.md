# Council Decision: Research Command Module

**Date:** February 7, 2026

**Proposal:** Add new `research` command module to AgentAssets for investigating and exploring feature ideas before implementation

---

## Proposal

**Title:** Add Research Command Module to AgentAssets

**Problem:**
AI agents need a structured approach for investigating and exploring feature ideas before committing to implementation. Currently, the `feature` command jumps directly to implementation planning without sufficient investigation of alternatives, assumptions, and feasibility.

**Evidence:**
- Feature command assumes problem is already validated
- No structured process for exploring alternatives
- Agents may implement features without thorough investigation
- Need to validate assumptions before committing resources

**Proposed Solution:**
Create new `research` command module based on the `feature` command structure but focused on investigation:
- INDEX.md - Overview and purpose
- PRESET.md - Research investigation template with 4-phase framework
- GUIDE.md - Step-by-step research workflow

**Alternatives Considered:**
1. Extend feature command with research phase - Rejected: Mixes two concerns
2. Add research section to refactor command - Rejected: Different purpose
3. Add research to council framework - Rejected: Council evaluates, doesn't research
4. Create standalone research command - Selected: Clean separation of concerns

**Costs:**
- LOC: ~300 lines of documentation
- Dependencies: None
- Testing: N/A (documentation)
- Maintenance: Low (markdown files)

**Benefits:**
- Structured investigation process before implementation
- Validates assumptions early
- Explores alternatives systematically
- Reduces wasted implementation effort
- Aligns with "best part is no part" by questioning necessity first

---

## Council Analysis

### Architect Analysis

**Structural Concerns:**

**Positive:**
- Clean separation of concerns (research vs. implementation)
- Follows established pattern (INDEX, PRESET, GUIDE)
- Fits naturally into AgentAssets structure
- Clear boundaries between research and feature commands

**Concerns:**
1. **Overlap with existing modules:** Some research content overlaps with feature/refactor
2. **Documentation structure:** 4-phase framework in PRESET.md is quite detailed (~200 lines)
3. **Transition clarity:** How does research hand off to feature command?
4. **Discovery:** How do agents know when to use research vs. feature?

**Structural Analysis:**
```
Current:
  commands/
    feature/     - Assumes problem validated
    refactor/    - Evaluates if refactor justified
    cleanup/     - Resource cleanup patterns

Proposed:
  commands/
    research/    - Investigate ideas before validation
    feature/     - Implement validated ideas
    refactor/    - Evaluate existing code changes
    cleanup/     - Resource cleanup patterns
```

This creates a clear progression: research → feature → implementation

**Impact on Architecture:**
- Adds new command type without changing existing ones
- Clear separation: research investigates, feature implements
- Fits naturally into existing patterns
- No coupling or dependency issues

**Vote: Approve with Conditions**

**Reasoning:**
Structure is sound and follows established patterns. The separation between investigation (research) and implementation (feature) is valuable. However, some refinement needed to reduce overlap and improve transition clarity.

**Conditions:**
1. Add clear transition guidance from research → feature
2. Reduce overlap with feature command (avoid duplicating content)
3. Add discovery guidance (when to use research vs. jumping to feature)
4. Consider simplifying PRESET.md (currently very detailed)

---

### Systems Analysis

**Infrastructure Concerns:**

**Positive:**
- No runtime code - pure documentation
- No dependencies added
- No resource usage
- No failure modes
- Zero performance impact

**Concerns:**
1. **Discoverability:** How do agents find/use this command?
2. **Integration:** How does this integrate with existing AgentAssets scripts?
3. **Sync:** Does auto_sync.py need to track research documents?
4. **Council integration:** Research findings should feed into council decisions

**Resource Analysis:**
- Disk space: ~15KB markdown files
- Memory: None (static documentation)
- CPU: None (static documentation)
- Impact: Zero

**Integration Points:**
- context_cli.py - May need to reference research module
- context_provider.py - Should provide research context when relevant
- decision_tracker.py - Could track research outcomes
- No integration required for core functionality

**Failure Modes:**
- None (documentation can't fail)
- Worst case: Agents don't use it (but no harm done)

**Vote: Approve**

**Reasoning:**
Zero infrastructure risk. Pure documentation with no runtime impact. No dependencies, no resource usage, no failure modes. From systems perspective, this is completely safe.

**Observations:**
Consider integrating research command into:
- Context provider for discovery
- Decision tracker for linking research to decisions
- But these are enhancements, not requirements

---

### UX Analysis

**User Experience Concerns:**

**Positive:**
- Structured approach reduces cognitive load
- Clear templates provide guidance
- Step-by-step workflow easy to follow
- Reduces risk of premature implementation

**Concerns:**
1. **Discovery:** How do agents know this command exists?
2. **When to use:** Unclear when to use research vs. feature command
3. **Length:** PRESET.md is very long (may be overwhelming)
4. **Workflow:** How does research integrate with existing workflow?
5. **Duplication:** Some content duplicates feature/refactor commands

**Workflow Analysis:**

Current workflow:
```
1. User requests feature
2. Agent uses @feature command
3. Agent implements directly
```

With research (unclear):
```
1. User requests feature (vague idea?)
2. Agent uses @research command (when?)
3. Agent investigates
4. Agent transitions to @feature (how?)
5. Agent implements
```

**Questions:**
- When should agent choose research vs. feature?
- Does user explicitly request research?
- Or does agent decide based on clarity of request?
- How does transition from research to feature work?

**Learning Curve:**
- Templates are comprehensive but long
- May be overwhelming for quick investigations
- Need lightweight quick-start option

**Consistency:**
- Follows same pattern as feature/refactor (good)
- INDEX/PRESET/GUIDE structure is familiar (good)
- Naming is clear and descriptive (good)

**Vote: Approve with Conditions**

**Reasoning:**
The research command adds value by providing structure for investigation, but needs clearer guidance on when/how to use it. The templates are comprehensive but may be too detailed for all use cases.

**Conditions:**
1. Add clear "When to Use" guidance in INDEX.md (done, but needs emphasis)
2. Add quick-start option for lightweight research
3. Clarify transition from research → feature
4. Consider splitting PRESET.md into basic and comprehensive templates
5. Add examples of good research vs. premature implementation

---

### Pragmatic Analysis

**Implementation Concerns:**

**Positive:**
- Quick to implement (markdown files)
- Low maintenance burden
- Easy to evolve based on usage
- Can gather feedback and refine
- Reversible (can delete if not useful)

**Concerns:**
1. **Usage:** Will agents actually use this?
2. **Overhead:** Does research command add too much process?
3. **Scope:** Is the framework too comprehensive?
4. **Practical value:** Does this solve a real problem?
5. **Adoption:** How do we encourage agents to use it?

**Complexity Analysis:**
- Implementation: Very simple (markdown files)
- Usage: Medium complexity (comprehensive templates)
- Maintenance: Low (static documentation)
- Testing: N/A (documentation)
- Total: LOW complexity

**MVP Analysis:**

What's the minimum viable research command?

**Current proposal (comprehensive):**
- INDEX.md with full overview
- PRESET.md with 4-phase framework (~200 lines)
- GUIDE.md with detailed methodology (~300 lines)
- Total: ~500 lines

**Minimal alternative:**
- INDEX.md with overview
- PRESET.md with basic research template (~50 lines)
- Skip GUIDE.md initially (can add later if needed)
- Total: ~100 lines

**Question:** Do we need the comprehensive version now, or start minimal and evolve?

**Practical Value:**

Does this solve a real problem?
- YES: Agents sometimes implement without investigating alternatives
- YES: Structured investigation reduces wasted effort
- YES: Validates assumptions before committing
- MAYBE: Could this be a section in feature command instead?

**Scope Evaluation:**

Is the current scope appropriate?
- 4-phase framework is comprehensive
- Research output template is detailed
- Step-by-step guide is thorough
- May be more than needed for most research

**Alternative: Start Minimal**
```markdown
# PRESET.md (Minimal)

## Research Questions Template

**What are we investigating?** [Clear question]

**What do we assume?** [Assumptions to validate]

**What are the alternatives?**
1. [Alternative 1]
2. [Alternative 2]
3. [Alternative 3]

**What did we learn?**
- [Finding 1]
- [Finding 2]

**What do we recommend?**
- [Proceed/Don't proceed]
- [Why]

**Next steps:**
- [ ] Action 1
- [ ] Action 2
```

Simple, clear, actionable. Can expand later if needed.

**Vote: Approve with Alternative**

**Reasoning:**
The research command is valuable but the current implementation is more comprehensive than needed for MVP. We should start minimal and expand based on actual usage patterns. This aligns with "simplicity and refinement" - start simple, refine based on feedback.

**Alternative: Minimal Viable Research Command**

**Phase 1 (Now):**
- INDEX.md - Brief overview
- PRESET.md - Simple research template (~50 lines)
- Skip GUIDE.md initially

**Phase 2 (After usage):**
- Expand PRESET.md based on what agents actually need
- Add GUIDE.md if detailed workflow is requested
- Add examples based on real research conducted

**Benefits of Minimal Approach:**
- Quick to implement
- Easy to understand
- Low barrier to adoption
- Can evolve based on actual needs
- Avoids over-engineering documentation

---

## Council Discussion

**Common Ground:**
- All members see value in research command
- All members agree separation from feature command is good
- All members want clearer discovery/transition guidance
- All members are concerned about comprehensiveness

**Key Disagreements:**
- **Scope:** Pragmatic wants minimal, Architect/UX want more structure
- **Completeness:** Trade-off between comprehensive templates vs. simple starting point

**Alternative Explored:**

**Minimal Viable Research Command:**
- Simple templates, clear guidance
- Can expand based on usage
- Aligns with "best part is no part"
- Easier to adopt and understand

**Compromise Position:**
- Keep comprehensive templates but add "Quick Start" section
- Provide both simple and detailed templates
- Users can choose based on needs
- Allows flexibility without overwhelming

**Consensus Approach:**
The council agrees to refactor the research command to provide both simple and comprehensive options, with clear guidance on when to use each.

---

## Unanimous Recommendation

**RECOMMENDATION: Proceed with Refinements**

The Council approves the research command module with specific refinements to improve usability and reduce overhead.

### Why This Is Valuable

**Architect Perspective:**
- Clean separation between investigation and implementation
- Fits naturally into existing structure
- Reduces risk of premature implementation

**Systems Perspective:**
- Zero infrastructure risk
- No dependencies or resource usage
- Safe to add and easy to remove if not useful

**UX Perspective:**
- Provides structure for investigation
- Reduces cognitive load with templates
- Helps agents make better decisions

**Pragmatic Perspective:**
- Low implementation cost
- Addresses real problem (premature implementation)
- Can evolve based on usage
- Reversible decision

### Required Refinements

#### 1. Simplify PRESET.md with Quick Start Options

Add at the beginning of PRESET.md:

```markdown
## Quick Start

### For Simple Research (5-15 minutes)
Use the **Basic Research Template** below for quick investigations.

### For Complex Research (1+ hours)
Use the **Comprehensive Research Framework** for thorough analysis.

---

## Basic Research Template

**Research Question:** [What are you investigating?]

**Assumptions:** [What do you assume is true?]

**Alternatives:**
1. [Alternative 1]
2. [Alternative 2]
3. [Alternative 3]

**Findings:**
- [Key finding 1]
- [Key finding 2]

**Recommendation:** [Proceed/Don't proceed and why]

**Next Steps:**
- [ ] Action 1
- [ ] Action 2

---

## Comprehensive Research Framework

[Keep existing 4-phase framework for complex research]
```

#### 2. Strengthen Discovery Guidance in INDEX.md

Add prominent section:

```markdown
## When to Use Research vs. Feature

**Use Research Command When:**
- Idea is vague or exploratory
- Multiple approaches possible
- Unclear if feature is needed
- Assumptions need validation
- Technical feasibility uncertain

**Use Feature Command When:**
- Problem is clearly defined
- Solution approach is evident
- Ready to plan implementation
- Research already completed

**Transition:** Research → findings → Feature → implementation
```

#### 3. Add Transition Guidance

Add to GUIDE.md:

```markdown
## Transitioning from Research to Feature

When research is complete and you've decided to proceed:

1. **Create Research Summary** - Distill key findings
2. **Reference in Feature Proposal** - Link research document
3. **Use Findings to Define Scope** - Research informs MVP
4. **Apply Learned Patterns** - Use research insights
5. **Validate During Development** - Test assumptions

**Template for Feature Proposal After Research:**
```
Feature Proposal: [Name]

Research Completed: [Link to research document]

Key Research Findings:
- [Finding 1 that informed this proposal]
- [Finding 2 that informed this proposal]

Validated Assumptions:
- [Assumption 1] ✓
- [Assumption 2] ✓

Recommended Approach: [Based on research alternative X]

[Continue with standard feature template]
```
```

#### 4. Add Examples Section

Create EXAMPLES.md:

```markdown
# Research Command Examples

## Example 1: Simple Research (Used Basic Template)

Research: Should we add keyboard shortcuts to timeline?

[Basic template filled out with findings]

Result: Proceeded to feature command

## Example 2: Complex Research (Used Comprehensive Framework)

Research: How should we implement real-time collaboration?

[Comprehensive framework filled out]

Result: Deferred - needs more technical investigation

## Example 3: Research Leading to "No"

Research: Should we add blockchain integration?

[Research template showing it's not needed]

Result: Did not proceed - no user need validated
```

### Success Criteria

**Adoption:**
- Agents use research command for exploratory work
- Clear decision-making on when to use research vs. feature
- Research findings inform better feature implementations

**Usability:**
- Simple template used for quick research (5-15 min)
- Comprehensive framework used for complex investigation (1+ hours)
- Clear guidance prevents confusion

**Value:**
- Reduces premature implementation
- Validates assumptions before commitment
- Explores alternatives systematically

### Action Items

- [ ] Add Quick Start section to PRESET.md with both simple and comprehensive templates
- [ ] Strengthen "When to Use" guidance in INDEX.md
- [ ] Add transition guidance to GUIDE.md
- [ ] Create EXAMPLES.md with real examples
- [ ] Update AgentAssets README.md to mention research command
- [ ] Consider adding to context_provider.py for discovery

### Implementation Plan

**Immediate (5-10 minutes):**
1. Refactor PRESET.md with Quick Start section
2. Update INDEX.md with discovery guidance
3. Add transition section to GUIDE.md

**Follow-up (after usage):**
1. Create EXAMPLES.md based on actual usage
2. Gather feedback on template usefulness
3. Refine based on what agents actually use
4. Consider integration with context_provider.py

### Why This Approach Is Better

**Flexibility:**
- Agents can choose simple or comprehensive based on needs
- No forced overhead for simple investigations
- Comprehensive option available for complex research

**Alignment with Core Values:**
- "Best part is no part" - Simple template is minimal
- "Simplicity and refinement" - Start simple, expand based on needs
- Pragmatic - Deliver value quickly, refine later

**Adoption:**
- Clear guidance reduces confusion
- Low barrier to entry with simple template
- Comprehensive option for when needed

**Sustainability:**
- Can evolve based on actual usage
- Not over-engineered upfront
- Easy to maintain and refine

---

## Conclusion

The research command is a valuable addition to AgentAssets that addresses a real need: structured investigation before implementation. With the recommended refinements, it provides flexibility for both quick investigations and comprehensive research while maintaining simplicity and clarity.

The council unanimously approves proceeding with the research command with the specified refinements.

**Next Step:** Implement refinements and observe usage patterns to guide future evolution.

---

*Council Members: Architect, Systems Engineer, UX Engineer, Pragmatic Engineer*
*Decision: Unanimous Approval with Refinements*
*Date: February 7, 2026*
