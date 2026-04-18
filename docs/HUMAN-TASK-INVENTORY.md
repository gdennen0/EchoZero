## Human Task Inventory

This is the working plan for full human-flow coverage in EchoZero.

Primary rule:

- launch only through `run_echozero.py`
- prefer semantic app control plus pointer-style input
- fall back to OS/system control only where the app does not own the UI

## Capability Layers

### Layer 1: Semantic app control

Use the app-owned automation surface for:

- visible targets
- actions
- selection state
- timeline hit targets
- screenshots
- sync state

### Layer 2: Pointer-style control

Use pointer primitives backed by semantic targets for:

- move pointer
- hover
- click
- double-click
- drag
- scroll

### Layer 3: OS fallback

Use system-level control for:

- native file dialogs
- focus changes
- packaged app smoke
- hardware/system-owned surfaces

## Task Matrix

### App lifecycle

- Launch app: covered
- New project: app-contract covered, semantic backend not yet stabilized
- Open project: app-contract covered, OS-dialog flow not yet covered
- Save project: app-contract covered, semantic backend save flow still needs hardening
- Save project as: app-contract covered, semantic backend save-as flow still needs hardening
- Close app with dirty-state handling: launcher covered

### Timeline/operator flow

- Import song: covered
- Extract stems: covered
- Extract drum events: covered
- Classify drum events: covered
- Select event: covered
- Nudge event: covered through scenario runner
- Duplicate event: covered through scenario runner
- Drag event: covered
- Scroll timeline: covered
- Hover/cursor movement: covered
- Double-click target: primitive covered, feature-specific flows still need expansion

### Transport

- Play: covered
- Pause: covered
- Stop: covered
- Seek by ruler/canvas interaction: partial
- Full hotkey coverage: partial

### Transfer/sync

- Enable sync: covered
- Disable sync: covered
- Push transfer surface: covered through scenario/app contract
- Pull transfer surface: covered through scenario/app contract
- Push/pull directly through semantic backend: not yet stable enough
- Live sync state branches: contract covered, human-flow coverage partial

### Versions and project structure

- Add song version: app-contract covered
- Switch song version: app-contract covered
- Layer creation/edit branches: partial
- Inspector action branches: partial

### System-owned flows

- Native file chooser interaction: not covered
- Packaged app automation: not covered
- Real MA3 hardware flow: not covered

## Execution Order

1. Stabilize semantic backend save/new/open flows.
2. Expand pointer primitives to all major target types.
3. Add direct semantic backend coverage for push/pull workspace branches.
4. Add OS-dialog fallback automation.
5. Add packaged-build automation lane.
6. Add real MA3 hardware/manual-assisted lane.

## Proof Commands

```bash
.venv/bin/python -m pytest tests/ui_automation/test_session.py tests/ui_automation/test_echozero_backend.py -q
.venv/bin/python -m echozero.testing.run --lane humanflow-all
```

## Live Attach

Start the real app with a localhost automation bridge:

```bash
.venv/bin/python run_echozero.py --automation-port 0
```

The launcher prints the bridge URL, for example:

```text
automation_bridge=http://127.0.0.1:43210
```

Current endpoints:

- `GET /health`
- `GET /snapshot`
- `POST /screenshot`
- `POST /action`

Current `POST /action` verbs:

- `move_pointer`
- `hover`
- `click`
- `double_click`
- `type_text`
- `press_key`
- `drag`
- `scroll`
- `invoke`
