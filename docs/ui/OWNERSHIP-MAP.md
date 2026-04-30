# UI Ownership Map

Status: reference
Last reviewed: 2026-04-30



This file tells humans and agents where UI concerns belong.
Use it before editing files that span UI, presentation, application, or shell
boundaries.

## Domain

- `Canonical Files`: `echozero/domain/*`
- `Allowed Here`: invariants, core entities, semantic correctness
- `Not Allowed Here`: widget behavior, surface policy, UI-facing display shaping

## Engine Or Core

- `Canonical Files`: `echozero/editor/*`, `echozero/pipelines/*`
- `Allowed Here`: execution mechanics, pipeline processing, staleness mechanics, app-agnostic processing behavior
- `Not Allowed Here`: EchoZero UI semantics, shell workflow rules, widget concerns

## Application

- `Canonical Files`: `echozero/application/timeline/*`, `echozero/application/sync/*`, `echozero/application/session/*`
- `Allowed Here`: intents, orchestration, side effects, truth mapping, workflow consequences, sync policy
- `Not Allowed Here`: widget paint/layout, local view heuristics, FEEL or token styling

## Presentation

- `Canonical Files`: `echozero/application/presentation/*`
- `Allowed Here`: typed UI-facing models, inspector contracts, derived display-ready structures
- `Not Allowed Here`: widget event capture, shell bootstrapping, business policy that belongs in application services

## UI Primitives

- `Canonical Files`: `echozero/ui/FEEL.py`, `echozero/ui/style/*`, `echozero/ui/qt/timeline/blocks/*`
- `Allowed Here`: tuning constants, visual tokens, geometry helpers, paint/layout primitives, reusable interaction helpers
- `Not Allowed Here`: application truth, project storage wiring, MA3 authority rules

## Runtime Shell

- `Canonical Files`: `echozero/ui/qt/app_shell.py`, `echozero/ui/qt/app_shell_*`
- `Allowed Here`: composition root, service wiring, runtime assembly, shell-level layout
- `Not Allowed Here`: business policy sink, duplicate truth shaping, widget-local semantic shortcuts promoted to shell policy

## Current High-Risk Drift Zones

- `echozero/ui/qt/timeline/widget.py`
  - Risk: viewport policy, gesture semantics, and duplicated labels living together
- `echozero/ui/qt/timeline/manual_pull.py`
  - Risk: duplicate timeline-like geometry and interaction rules
- `echozero/ui/qt/timeline/object_info_panel.py`
  - Risk: partial generic inspector, partial hard-coded action panel
- `echozero/ui/qt/app_shell.py`
  - Risk: composition root mixed with projection and runtime policy

## Working Rule

If a change touches truth, interaction semantics, surface responsibilities, and
rendering at the same time:

1. change the application or presentation contract first
2. update the relevant inventory
3. change the widget or shell second
