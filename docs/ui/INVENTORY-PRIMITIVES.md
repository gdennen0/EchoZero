# Primitive Inventory

Status: reference
Last reviewed: 2026-04-30



This file is the canonical inventory of reusable UI and presentation
primitives.
Prefer promoting real seams already present in the repo over inventing new
meta-structure.

## Entry Format

- `Name`
- `Status`
- `Purpose`
- `Owner Layer`
- `Canonical Files`
- `Used By`
- `Notes`

Statuses:

- `canonical`
- `candidate`
- `deprecated`

## Presentation Models

- `Name`: Presentation Models
- `Status`: canonical
- `Purpose`: Typed UI-facing data structures for timeline, rows, actions, and display state.
- `Owner Layer`: presentation
- `Canonical Files`: `echozero/application/presentation/models.py`
- `Used By`: assembler, widget, object info panel
- `Notes`: Promote this as the source for reusable UI-facing truth, not widget-local shaping.

## Inspector Contract

- `Name`: Inspector Contract
- `Status`: canonical
- `Purpose`: Typed facts, sections, and actions for inspector-like surfaces.
- `Owner Layer`: presentation
- `Canonical Files`: `echozero/application/presentation/inspector_contract.py`
- `Used By`: object info panel, transfer-related UI
- `Notes`: Good candidate for a neutral inspector primitive across more than timeline.

## Timeline Assembler

- `Name`: Timeline Assembler
- `Status`: candidate
- `Purpose`: Shape application truth into display-ready presentation.
- `Owner Layer`: application/presentation boundary
- `Canonical Files`: `echozero/application/timeline/assembler.py`
- `Used By`: timeline app and runtime shell
- `Notes`: Should absorb duplicated header and label shaping now living elsewhere.

## FEEL Constants

- `Name`: FEEL Constants
- `Status`: canonical
- `Purpose`: Own interaction and layout tuning constants.
- `Owner Layer`: UI primitives
- `Canonical Files`: `echozero/ui/FEEL.py`
- `Used By`: timeline widget, blocks, future surfaces
- `Notes`: No magic-number drift outside FEEL-backed or style-backed surfaces.

## Style Tokens And Builders

- `Name`: Style Tokens And Builders
- `Status`: canonical
- `Purpose`: Own visual tokens, scales, and QSS building.
- `Owner Layer`: UI primitives
- `Canonical Files`: `echozero/ui/style/tokens.py`, `echozero/ui/style/scales.py`, `echozero/ui/style/qt/qss.py`, `echozero/ui/qt/timeline/style.py`
- `Used By`: shell and timeline surfaces
- `Notes`: Visual semantics belong here rather than in widgets.

## Timeline Paint And Layout Blocks

- `Name`: Timeline Paint And Layout Blocks
- `Status`: canonical
- `Purpose`: Stateless or presentation-driven timeline rendering primitives.
- `Owner Layer`: UI primitives
- `Canonical Files`: `echozero/ui/qt/timeline/blocks/*`
- `Used By`: timeline workspace
- `Notes`: Prefer reusing these blocks over widget-local custom paint logic.

## Time-Axis Math

- `Name`: Time-Axis Math
- `Status`: candidate
- `Purpose`: Shared mapping between time, viewport span, x-position, and scroll behavior.
- `Owner Layer`: UI primitives
- `Canonical Files`: `echozero/ui/qt/timeline/blocks/ruler.py`, `echozero/ui/qt/timeline/blocks/waveform_lane.py`, `echozero/ui/qt/timeline/widget.py`
- `Used By`: timeline workspace, manual pull UI
- `Notes`: High-value extraction candidate; currently still split across several files.

## Projection And Label Helpers

- `Name`: Projection And Label Helpers
- `Status`: candidate
- `Purpose`: Build canonical labels, summaries, badges, and timeline projection helpers from storage-backed data.
- `Owner Layer`: application/presentation boundary
- `Canonical Files`: `echozero/ui/qt/app_shell_project_timeline.py`
- `Used By`: runtime shell, timeline view path
- `Notes`: Promote these rather than duplicating labels in widget/demo code.
