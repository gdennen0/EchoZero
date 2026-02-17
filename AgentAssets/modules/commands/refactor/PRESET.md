# REFACTOR PROPOSAL

**Command:** "Propose refactor for [area]"

**Purpose:** Evaluate if refactoring is justified.

---

## QUESTIONS (Must Answer)

1. What concrete problem does this solve?
2. Is there a simpler fix?
3. **Can we DELETE instead of reorganize?**
4. Net complexity change?
5. Real problem or imagined?

---

## RED FLAGS (Reject)

- "More flexible"
- "Cleaner"
- "Best practices"
- No concrete problem
- "Might need later"

---

## GREEN FLAGS (May Proceed)

- Pattern emerged 3+ times
- Clear bugs from current structure
- Fewer lines after
- Deletion opportunity

---

## OUTPUT

```
REFACTOR: [Area]

Problem: [Specific, concrete]

Change: [What]

Before: [Brief]
After: [Brief]

Net: +X / -Y lines
Risk: [Assessment]
```

---

## RULES

- Require evidence of actual pain
- Prefer deletion
- Small incremental changes
- Don't refactor while fixing bugs

