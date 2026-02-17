---
name: echozero-council
description: EchoZero Council decision framework for evaluating proposals. Use when evaluating major features, architectural changes, refactors, or when the user asks to assemble the council, get council review, or evaluate a proposal against core values.
---

# Council Decision Framework

## Council Lenses

Assemble Architect, Systems, UX, and Pragmatic Engineer. Each analyzes from their lens:

| Lens | Focus |
|------|-------|
| Architect | Structure, boundaries, coupling, abstraction level |
| Systems | Stability, resources, performance, failure modes |
| UX | Usability, discoverability, error messages, mental model |
| Pragmatic | Implementation complexity, testing, scope, maintenance |

## Core Value Filters

1. **Necessity:** Real problem? Evidence of need? Can we remove instead?
2. **Simplicity:** Simplest solution? 80/20? Current vs imagined needs?
3. **Cost/Benefit:** LOC, dependencies, testing, maintenance vs benefits
4. **Alignment:** Fits "best part is no part"? Consistent with architecture?

## Vote Options

- **Approve** - Passes all filters
- **Approve with Conditions** - Fixable issues, clear criteria
- **Reject with Alternative** - Fundamental issues, better approach exists
- **Reject** - Violates values, problem doesn't need solving

## Analysis Format

```
[Lens] Analysis:
Problem Understanding: [What's being solved]
Key Concerns: [From your lens]
Alternatives: [Considered]
Vote: [Approve/Approve w/ Conditions/Reject w/ Alternative/Reject]
Reasoning: [Explanation]
```

## Red Flags (Reject)

- Architect: "Might need flexibility later", >3 abstraction layers
- Systems: "Should be fine", no error handling
- UX: "Users will figure it out", cryptic errors
- Pragmatic: "Build it right", can't define MVP

## Green Flags (Approve)

- Architect: Removes abstractions, reduces coupling
- Systems: Clear resource management, explicit errors
- UX: Common case easier, self-explanatory
- Pragmatic: Simple to implement, easy to test

## Output Format

```
RECOMMENDATION: [Proceed / Proceed with Modifications / Use Alternative / Defer / Reject]

[Consensus reasoning]
Conditions (if any): [...]
Next Steps: [...]
```

## Reference

Full framework: `AgentAssets/modules/process/council/FRAMEWORK.md`
