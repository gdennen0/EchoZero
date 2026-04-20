# UI Change Checklist

_Updated: 2026-04-18_

Use this checklist for meaningful UI or architecture work.
Keep it lightweight.
If most fields are irrelevant, the change is probably too small to require a
full pass.

## Required Questions

- `Concern Type`:
  - state / interaction / surface / language / primitive / feedback
- `Owner Layer`:
- `Canonical Pattern Reused`:
- `New Pattern Introduced`:
- `Files Changed`:
- `Inventory Or Doc Updates Required`:
- `Proof Lane`:
- `Risks`:

## Hard Update Triggers

- New interaction -> update `docs/ui/INVENTORY-INTERACTIONS.md`
- New surface role or recurring sub-surface pattern -> update `docs/ui/INVENTORY-SURFACES.md`
- New reusable primitive or promoted reusable seam -> update `docs/ui/INVENTORY-PRIMITIVES.md`
- Ownership drift or file-boundary clarification -> update `docs/ui/OWNERSHIP-MAP.md`
- Standard or rule change -> update `docs/ui/STANDARDS.md`

## Preferred Patterns

- typed app-backed state
- derived presentation
- canonical intents
- shared primitives
- inventory-backed patterns

## Avoid

- widget-local semantics
- duplicated mutable state
- manual UI fan-out refresh wiring
- ad hoc labels or status logic
- one-off interactions
- shell-owned business policy

## Review Rule

If implementation changes a canonical UI pattern, update the relevant inventory
in the same task or PR.
