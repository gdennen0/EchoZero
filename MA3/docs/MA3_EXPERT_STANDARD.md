# MA3 Expert Standard

Status: active
Last updated: 2026-05-03

This document defines the standard EchoZero path for building MA3 expertise
that both humans and LLM agents can rely on.

## Core Split

There are two different surfaces and they must not be conflated:

1. MA terminal/CLI is the raw grandMA3 authority.
2. OSC is the EchoZero custom service/API layer we build on top.

The MA terminal/CLI exists so we can inspect the console directly and access
all native MA3 information without filtering it through our own Lua API.

OSC exists so EchoZero can call stable custom functions and workflows that we
choose to expose.

## Goal

MA3 knowledge must come from three layers that stay distinct:

1. Official MA documentation for syntax, API shape, and command semantics.
2. Live capture artifacts from our custom OSC service layer for EchoZero-facing
   browse/query workflows.
3. EchoZero learnings for guardrails, quirks, and proven integration patterns.

Hand-written notes are useful, but they are not the authority once direct MA
terminal evidence exists.

## Canonical Source Stack

Official MA docs:

- Lua object-free API:
  `https://help.malighting.com/grandMA3/2.3/HTML/lua_objectfree.html`
- Native syntax keywords:
  `https://help.malighting.com/grandMA3/2.3/HTML/csk_general_keywords.html`
- Extended command line options:
  `https://help.malighting.com/grandMA3/2.3/HTML/extended_command_line.html`
- Parent/child syntax:
  `https://help.malighting.com/grandMA3/2.3/HTML/csk_parent_child.html`
- Manual root:
  `https://help.malighting.com/grandMA3/2.3/HTML/help.html`

EchoZero repo docs:

- `MA3/README.md`
- `MA3/docs/MA3_LUA_API_REFERENCE.md`
- `MA3/docs/MA3_LEARNINGS.md`
- `MA3/docs/MA3_SEQUENCE_MANAGEMENT.md`
- `MA3/docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md`

Live capture artifacts:

- `artifacts/ma3-datapool/**`

Generated artifacts are evidence, not repo source. Do not commit them.

## Locked Rule

Do not treat OSC as the raw MA authority.

That means the following must be captured through the MA terminal/CLI surface as
native proof:

- `Dump()`
- `PropertyCount()`
- `PropertyName()`
- `PropertyType()`
- `PropertyInfo()`
- native `List` / `ChangeDestination` inspection output

On the live MA3 terminal, treat the property APIs as handle methods in practice:

- `DataPool()[6][1]:PropertyCount()`
- `DataPool()[6][1]:PropertyName(0)`
- `DataPool()[6][1]:PropertyType(0)`
- `DataPool()[6][1]:PropertyInfo(0)`

## Standard Capture Workflow

Capture the full DataPool tree from the live MA3 target through our custom OSC
layer when you want EchoZero-facing browse structure:

```bash
./.venv/bin/python MA3/dev/ma3_datapool_documenter.py \
  --ma3-host 127.0.0.1 \
  --ma3-port 8001 \
  --output-dir artifacts/ma3-datapool/latest
```

Capture one subtree when a task is narrower:

```bash
./.venv/bin/python MA3/dev/ma3_datapool_documenter.py \
  --ma3-host 127.0.0.1 \
  --ma3-port 8001 \
  --root-path Timecodes \
  --output-dir artifacts/ma3-datapool/timecodes
```

The documenter writes:

- `snapshot.json`: machine-readable hierarchy inventory from our custom OSC browse
- `hierarchy.md`: path-first hierarchy view
- `terminal_capture_plan.md`: per-object MA terminal proof checklist
- `terminal_capture_targets.json`: machine-readable MA terminal capture manifest

Build the native class/property catalog from a sample crawl:

```bash
./.venv/bin/python MA3/dev/ma3_terminal_class_catalog.py \
  --host 127.0.0.1 \
  --sample-json artifacts/ma3-terminal-crawl/sample/datapool_sample_crawl.json \
  --output-dir artifacts/ma3-terminal-crawl/class-catalog/latest
```

This writes:

- `node_property_inventory.json`: per-object native property metadata
- `class_catalog.json`: machine-readable class/property schema summary
- `class_catalog.md`: human-readable class/property catalog
- `preset_pool_notes.md`: preset-focused property notes from the same pass

## Standard Object Truth

For hierarchy and attributes, use the generated bundle in this order:

1. `hierarchy.md` to navigate paths and parent/child relationships
2. `terminal_capture_plan.md` to drive native MA inspection
3. terminal output artifacts as the final authority for raw MA attributes
4. `snapshot.json` for machine-readable EchoZero-side hierarchy automation

If a hand-written doc disagrees with fresh terminal output, the terminal output
wins for that MA3 version and show state.

## Crawl Efficiency Rule

Do not brute-force every instance in large uniform pools unless variance is the
actual question.

Default approach:

- discover hierarchy broadly
- group objects by class
- sample representative objects for native attribute inventory
- escalate to full per-instance capture only when a class shows meaningful
  schema or child-structure variance

Example:

- `Sequences` usually do not need full per-instance dump coverage just to learn
  the class attributes

## Plugin Contract

The MA3 OSC plugin/service layer must expose enough structure for EchoZero docs
and automation:

- generic DataPool child traversal
- generic object description by path

This custom layer is useful and intentional, but it must not be treated as the
source of truth for native dump or property inventory.

This is now the standard for any future MA3 browse or introspection work.

## Limits

- DataPool captures are show-state dependent.
- Native property inventory can change across MA3 versions.
- A capture is authoritative for the target it came from, not for every show.
- EchoZero policy docs still own product behavior on top of MA3 mechanics.
