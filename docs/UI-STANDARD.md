# UI Standard

Status: active canonical UI standard
Last verified: 2026-04-21

_Updated: 2026-04-16_

This document defines the North Star for EchoZero UI design and implementation.
It exists to give the repo one stable standard for desktop UI decisions before
and during UI engine redevelopment.
It connects Qt implementation constraints, Fluent-style system thinking, FEEL,
and EchoZero operator workflow rules into one design standard.

For the current canonical UI governance layer, also read:

- `docs/ui/STANDARDS.md`
- `docs/ui/INVENTORY-INTERACTIONS.md`
- `docs/ui/INVENTORY-SURFACES.md`
- `docs/ui/INVENTORY-PRIMITIVES.md`
- `docs/ui/OWNERSHIP-MAP.md`
- `docs/ui/CHANGE-CHECKLIST.md`

## North Star

EchoZero uses this standard:

- **General UI system:** Fluent 2
- **Implementation model:** tokenized Qt Widgets styling plus custom-painted domain surfaces
- **EchoZero extension:** timeline/canvas/workspace rules for audio and show-control workflows

This is not a request to clone Microsoft UI literally.
It is a request to adopt a real system standard for:

- tokens
- hierarchy
- spacing
- states
- typography roles
- panel structure
- restraint

Then extend it only where EchoZero has domain-specific surfaces that a standard
desktop design system does not cover.

## Why This Standard

EchoZero is a desktop productivity tool, not a marketing site and not a
phone-first app.

So the standard should optimize for:

- calm, readable operator surfaces
- dense but understandable information
- strong keyboard/mouse affordance
- stable panel/workspace hierarchy
- scalable token-based styling
- custom-rendered timeline and waveform surfaces

Fluent 2 fits that better than web-marketing-first systems.
Qt Widgets styling also maps naturally to a token-and-QSS approach.

## Qt Implementation Rules

### Standard widgets

For standard desktop controls and shell surfaces:

- use standard Qt widgets where standard desktop controls already exist
- use tokenized `QSS`
- use shared shell tokens and scales
- prefer reusable style builders over local inline styling

### Custom domain surfaces

For timeline, waveform, ruler, playhead, lane, and other canvas-heavy surfaces:

- use custom painting
- keep painting stateless where possible
- feed paint/layout primitives with typed presentation data
- keep business truth out of paint code

### Accessibility and input

Desktop accessibility and keyboard flow are first-class requirements.

That means:

- complete and intentional tab/focus order
- keyboard shortcuts for high-frequency actions
- visible focus behavior
- accessible names/roles where applicable
- no essential interaction hidden behind hover-only discovery

### Do not do this by default

- do not build the whole system around a custom `QStyle`
- do not scatter styling constants through widgets
- do not mix visual token rules with business semantics

## Core Principles

### 1. System before screen

Before building a new surface, define:

- visual thesis
- workspace/content plan
- interaction thesis

No substantial UI task should begin with ad hoc widget assembly.

### 2. Calm operator surfaces

Default app UI should be:

- calm
- dense but readable
- low-chrome
- high-signal
- scan-friendly

If a user can understand the screen by scanning headings, labels, numbers, and
primary controls, the surface is doing its job.

Keyboard and pointer efficiency matter more than decorative novelty.

### 3. One job per region

Each panel, section, or area should have one main responsibility:

- orient
- inspect
- edit
- navigate
- confirm
- monitor

If a region is trying to do two or three things, split or simplify it.

### 4. Layout before cards

Cards are not the default.

Use:

- spacing
- grouping
- dividers
- panel boundaries
- alignment
- typography

before adding boxed surfaces.

Cards are allowed only when the card itself is the interaction or the boundary
materially improves comprehension.

### 5. Utility copy over marketing copy

For operator/product surfaces:

- headings should identify the area or action
- supporting copy should explain state, behavior, or consequence
- avoid aspirational or homepage-style copy

EchoZero is an operating surface, not a landing page.

### 6. FEEL owns tuning

Interaction and timeline tuning constants belong in:

- `echozero/ui/FEEL.py`

Do not spread timing, dimensions, or interaction thresholds across unrelated
widget code.

### 7. Truth stays in application contracts

The UI may render truth and route interaction, but it must not invent truth.

That means:

- no widget-only workflow semantics
- no title/badge/label heuristics standing in for typed state
- no resurrection of active-take truth behavior
- no UI-only sync truth

### 8. High-DPI and resize stability

The UI must remain stable across:

- different desktop scaling factors
- different monitor densities
- resizing and panel reflow

Custom-painted surfaces should use scale-aware geometry, icons, and text sizing.

## Standard Surface Model

EchoZero desktop UI should generally organize into:

- primary workspace
- navigation
- secondary context / inspector
- one clear accent for action or state

This matches the kind of app structure where Fluent-style restraint works well.

For timeline/editor views, the extension layer adds:

- ruler
- transport
- lane stack
- take rows
- waveform/event rendering
- inspector-backed detail

## Token Model

The system should be expressed in tokens first.

Minimum token categories:

- background
- surface
- panel
- border
- primary text
- secondary text
- accent
- disabled state
- focus/selection state
- spacing scale
- radius scale
- typography roles

Preferred implementation homes:

- `echozero/ui/style/tokens.py`
- `echozero/ui/style/scales.py`
- `echozero/ui/style/qt/qss.py`
- `echozero/ui/qt/timeline/style.py`

## Typography Rules

- strong type hierarchy
- few typefaces
- predictable role names
- readable dense UI over decorative type treatments

Default roles:

- display
- headline
- section title
- body
- secondary/meta
- caption/label

Do not improvise font sizing widget by widget.

## Color Rules

- few colors
- one clear accent by default
- state colors must be semantically legible
- decorative gradients are not a substitute for hierarchy

Routine operator UI should not depend on loud gradients or ornamental color use.

## Motion Rules

Motion is allowed only when it improves:

- hierarchy
- affordance
- orientation
- transition clarity

Default rule:

- ship a few intentional motions, not many decorative ones

Do not add motion that exists only to make the UI feel “fancy”.

## Desktop-Specific Rules

What we adopt from broader UI guidance:

- strong constraints before implementation
- explicit token system
- calm hierarchy
- real content
- restrained color and chrome
- one job per region

What we adapt for desktop Qt:

- use QSS/tokens for standard widgets
- use custom paint for timeline/canvas surfaces
- prioritize panel hierarchy, focus behavior, keyboard affordance, resize stability, scanability, and high-DPI correctness

What we ignore as web-specific by default:

- hero-section rules
- landing-page narrative sequencing
- web-only animation stacks
- homepage-style brand theatrics on operator surfaces

## Reusable Library Direction

The reusable internal UI library should focus on:

- theme/tokens
- shell surface scaffolding
- geometry helpers
- time-axis helpers
- generic paint helpers

The following should stay EchoZero-specific unless a second real consumer exists:

- timeline truth semantics
- inspector action semantics
- MA3 transfer semantics
- take-row meaning
- provenance/status interpretation

## Workflow Rule For Agents

Before substantial UI work, define:

- visual thesis
- workspace/content plan
- interaction thesis
- proof lane

If those are missing, the task is under-specified.

## Review Checklist

When evaluating a UI change, ask:

1. Is this using the token system or inventing local styling?
2. Does this surface read clearly by scanning headings, labels, and numbers?
3. Is a card really necessary here?
4. Is any business truth being invented in the widget layer?
5. Is the keyboard/focus/accessibility behavior still strong?
6. Does the change improve the reusable engine or just add more local scaffolding?
7. What app-path proof confirms it?

## Sources

This standard is primarily informed by:

- Qt Widgets styling docs:
  - https://doc.qt.io/qtforpython-6.9/overviews/qtwidgets-qwidget-styling.html
  - https://doc.qt.io/qtforpython-6/overviews/qtwidgets-stylesheet.html
  - https://doc.qt.io/qtforpython-6.5/PySide6/QtWidgets/QStyle.html
  - https://doc.qt.io/qt-6/accessible-qwidget.html
- Fluent 2 design principles and tokens:
  - https://fluent2.microsoft.design/design-principles
  - https://fluent2.microsoft.design/design-tokens
- Desktop accessibility and interaction references:
  - https://learn.microsoft.com/en-us/windows/apps/design/accessibility/keyboard-accessibility
  - https://learn.microsoft.com/en-us/windows/apps/design/input/access-keys
- OpenAI frontend guidance, adapted for desktop/operator surfaces:
  - https://developers.openai.com/blog/designing-delightful-frontends-with-gpt-5-4
