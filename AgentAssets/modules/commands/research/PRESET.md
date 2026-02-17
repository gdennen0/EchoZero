# Research Investigation Preset

## Quick Start

### For Simple Research (5-15 minutes)
Use the **Basic Research Template** below for quick investigations, exploratory questions, or validating simple ideas.

### For Complex Research (1+ hours)
Use the **Comprehensive Research Framework** for thorough analysis of major features, architectural decisions, or complex problems.

---

## Basic Research Template

**Research Question:** [What are you investigating?]

**Why:** [Why investigate this now?]

**Assumptions:** [What do you assume is true?]

**Alternatives:**
1. [Alternative 1]
2. [Alternative 2]
3. [Alternative 3]

**Key Findings:**
- [Finding 1]
- [Finding 2]
- [Finding 3]

**Recommendation:** [Proceed/Don't proceed/Need more info]

**Reasoning:** [Why this recommendation?]

**Next Steps:**
- [ ] Action 1
- [ ] Action 2

---

## Comprehensive Research Framework

### Research Proposal Template

**Research Topic:** [Clear description of what you're investigating]

**Initial Hypothesis:** [What do you think might be true or needed?]

**Research Questions:**
1. [Question 1 - What do you need to know?]
2. [Question 2]
3. [Question 3]

**Context:**
- **Problem Space:** [What problem area are you exploring?]
- **Current State:** [What exists now? What's missing?]
- **Motivation:** [Why investigate this now?]

---

## Investigation Framework

### Phase 1: Problem Exploration

**1. Define the Problem Space**
- What problem are we trying to understand?
- Who is affected by this problem?
- What evidence do we have that this is a real problem?
- What happens if we don't solve it?

**2. Question Assumptions**
- What assumptions are we making?
- Can we validate these assumptions?
- What would prove our assumptions wrong?
- Are we solving the right problem?

**3. Explore Existing Solutions**
- What already exists in EchoZero?
- What exists in similar systems?
- What patterns are already established?
- Can existing solutions be extended?

### Phase 2: Technical Investigation

**1. Architecture Analysis**
- Where would this fit in the current architecture?
- What layers would be affected?
- Are there architectural constraints?
- What dependencies would be needed?

**2. Feasibility Research**
- Is this technically feasible?
- What are the technical challenges?
- What are the performance implications?
- What are the maintenance costs?

**3. Pattern Research**
- Are there existing patterns we can use?
- What patterns exist in similar domains?
- What anti-patterns should we avoid?
- How do similar systems solve this?

### Phase 3: Alternative Exploration

**1. Generate Alternatives**
- What are different ways to solve this?
- What are simpler approaches?
- Can we solve this without building anything?
- Can we remove something instead?

**2. Compare Approaches**
- What are the trade-offs?
- What's the complexity of each?
- What's the maintenance burden?
- What's the user experience impact?

**3. Evaluate Against Core Values**
- Which approach aligns with "best part is no part"?
- Which is simplest?
- Which reduces complexity?
- Which can be deleted later if not needed?

### Phase 4: Validation

**1. Validate Assumptions**
- What did we learn that confirms assumptions?
- What did we learn that contradicts assumptions?
- What new questions emerged?
- What risks did we identify?

**2. Gather Evidence**
- What evidence supports proceeding?
- What evidence suggests not proceeding?
- What information is still missing?
- What would we need to know to decide?

**3. Document Findings**
- What are the key insights?
- What are the recommendations?
- What are the open questions?
- What are the next steps?

---

## Research Output Template

```
RESEARCH: [Topic]

Initial Hypothesis: [What you thought]

Research Questions:
1. [Question 1]
2. [Question 2]
3. [Question 3]

Findings:

Problem Space:
- [Finding 1]
- [Finding 2]

Technical Investigation:
- [Finding 1]
- [Finding 2]

Alternatives Explored:
1. [Alternative 1] - [Pros/Cons]
2. [Alternative 2] - [Pros/Cons]
3. [Alternative 3] - [Pros/Cons]

Key Insights:
- [Insight 1]
- [Insight 2]

Validated Assumptions:
- [Assumption 1] - [Validated/Contradicted]
- [Assumption 2] - [Validated/Contradicted]

Open Questions:
- [Question 1]
- [Question 2]

Recommendations:
- [Recommendation 1]
- [Recommendation 2]

Next Steps:
- [ ] [Action 1]
- [ ] [Action 2]
- [ ] Transition to feature command if proceeding
```

---

## Research Checklist

### Exploration Phase
- [ ] Problem space clearly defined
- [ ] Assumptions identified and questioned
- [ ] Existing solutions researched
- [ ] Evidence gathered about problem

### Investigation Phase
- [ ] Architecture implications analyzed
- [ ] Technical feasibility assessed
- [ ] Patterns researched
- [ ] Dependencies identified

### Alternative Phase
- [ ] Multiple alternatives explored
- [ ] Trade-offs evaluated
- [ ] Approaches compared against core values
- [ ] Simplest approach identified

### Validation Phase
- [ ] Assumptions validated or contradicted
- [ ] Evidence gathered
- [ ] Findings documented
- [ ] Recommendations made
- [ ] Next steps determined

---

## Red Flags (Reconsider Research Direction)

- Researching without a clear question
- Assuming the solution before understanding the problem
- Ignoring existing solutions
- Not questioning assumptions
- Researching features that don't solve real problems
- Skipping alternative exploration

---

## Green Flags (Good Research)

- Clear research questions
- Assumptions explicitly identified
- Existing solutions explored
- Multiple alternatives considered
- Evidence-based findings
- Recommendations grounded in research
- Alignment with core values

---

## Transition to Feature Development

When research is complete and you've decided to proceed:

1. Use findings to inform feature proposal
2. Reference research in feature PRESET.md
3. Use research insights to define MVP scope
4. Apply learned patterns in implementation
5. Validate research assumptions during development

See: `modules/commands/feature/` for implementation workflow
