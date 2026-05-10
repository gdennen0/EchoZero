# MA3 Plugins

This directory contains grandMA3 Lua plugins for EchoZero integration.

## Quick Start for Developers

1. **API Reference**: `docs/MA3_LUA_API_REFERENCE.md` - Complete Lua API organized by purpose
2. **Learnings**: `docs/MA3_LEARNINGS.md` - Practical gotchas and patterns discovered
3. **Sequence Prep**: `docs/MA3_SEQUENCE_MANAGEMENT.md` - MA3-side sequence listing, creation, assignment, and track prep contract
4. **Expert Standard**: `docs/MA3_EXPERT_STANDARD.md` - Canonical source stack and DataPool capture standard
5. **Plugin Guide**: `docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md` - Full development guide

## Directory Structure

```
MA3/
  README.md           # This file
  MA3_INTEGRATION_PITFALLS.md  # Critical gotchas (read first)
  docs/               # Documentation
  plugins/            # Production plugins (echozero_spine)
  dev/                # Development/exploration tools
```

## Production Plugin

The current EchoZero MA3 plugin is a small bundle of Lua modules.

- **Repo source**: `MA3/plugins/`
- **Active live bundle on this machine**:
  `/Users/march/MALightingTechnology/gma3_library/datapools/plugins/EZ/`
- **Core modules**:
  - `echozero.lua` / `ez_core.lua` for the main EZ API
  - `timecode.lua` for browse/query helpers
  - `Sequence.lua` / `ez_sequence.lua` for sequence prep helpers
  - `echozero_osc.lua` / `ez_osc.lua` for OSC transport
  - `echozero_init.lua` / `ez_init.lua` for startup verification

The live MA3 bundle may use `ez_*` filenames while the repo source still uses
the older `echozero_*` split. Keep behavior aligned when editing.

**Key Functions:**
```lua
-- Connection
EZ.Ping()                           -- Test connection
EZ.SetTarget(ip, port)              -- Set EchoZero address
EZ.Version()                        -- Print EZ version/build in MA3 feedback

-- Queries
EZ.GetTimecodes()                   -- List all timecodes
EZ.GetTrackGroups(tcNo)             -- List track groups
EZ.GetTracks(tcNo, tgNo)            -- List tracks
EZ.GetEvents(tcNo, tgNo, trackNo)   -- Get track events
EZ.GetSequences(startNo, endNo)     -- List sequences, optionally filtered by range
EZ.GetCurrentSongSequenceRange()    -- Resolve SpeedOfLight current-song range

-- Manipulation
EZ.CreateTrack(tcNo, tgNo, name)    -- Create new track
EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd[, eventName[, cueNo[, cueLabel[, channelNo[, explicitEventName[, explicitCueName]]]]]])  -- Add event
EZ.UpdateEventMeta(tcNo, tgNo, trackNo, idx, eventName, cueNo, cueLabel)       -- Update event name/cue metadata
EZ.DeleteEvent(tcNo, tgNo, trackNo, idx)     -- Delete event
EZ.CreateSequenceNextAvailable(name)         -- Create next free MA3 sequence
EZ.CreateSequenceInCurrentSongRange(name)    -- Create sequence in current-song range
EZ.AssignTrackSequence(tcNo, tgNo, trackNo, seqNo) -- Assign sequence to track
EZ.PrepareTrackForEvents(tcNo, tgNo, trackNo)      -- Ensure TimeRange/CmdSubTrack

-- Transport / Playhead
EZ.Play([tcNo])                                    -- Trigger EZ transport play state
EZ.Pause([tcNo])                                   -- Trigger EZ transport pause state
EZ.Stop([tcNo])                                    -- Trigger EZ transport stop state
EZ.JumpToPreviousSection([tcNo[, tgNo, trackNo]])  -- Jump to previous section marker/event
EZ.JumpToNextSection([tcNo[, tgNo, trackNo]])      -- Jump to next section marker/event
EZ.ScrubTimecodeBy(deltaSeconds[, tcNo])           -- Scrub by signed seconds (+/-)
EZ.ScrubTimecodeForward([stepSeconds[, tcNo]])     -- Scrub forward (default 1s)
EZ.ScrubTimecodeBackward([stepSeconds[, tcNo]])    -- Scrub backward (default 1s)

-- Real-time Sync
EZ.HookTrack(tcNo, tgNo, trackNo)   -- Start monitoring
EZ.UnhookTrack(tcNo, tgNo, trackNo) -- Stop monitoring
EZ.UnhookAll()                      -- Remove all hooks
```

## EchoZero Cue Semantics v0

For the current EchoZero MA3 push/pull lane, keep cue semantics intentionally narrow:

- Push writes one-shot `Go` / `Goto` style command cues at the EchoZero event start time.
- Pull reads MA3 events as one-shot command cues at their MA3 event time.
- EchoZero does not yet model temp press/unpress behavior, paired off cues, or richer MA3 sequence semantics.
- When MA3 provides event duration, EchoZero preserves it. When MA3 only provides event time, EchoZero imports a short one-shot event at that start time.

This is a deliberate v0 contract for reliable sequence/timecode integration, not a claim that MA3 only supports this cue model.

## Current App Contract (2026-04-23)

- Push routing truth remains `EZ layer -> MA3 track coord`.
- Push can prepare empty MA3 targets by assigning an existing sequence or creating a new one before write.
- Active `SongVersion` records carry one MA3 timecode pool number.
- New songs receive the next unused project-local MA3 timecode pool by default.
- New versions inherit the source song version's MA3 timecode pool.
- Pull defaults to the selected layer route when present; otherwise it defaults to the active song version's MA3 timecode pool.
- Pull workspace browsing is TC-first: pick the MA3 timecode pool, then view every track group and track for that pool in one clickable workspace.
- Pull import mode is destination-driven: new EZ layer targets import to `main`, while existing EZ layer targets import as a new take.
- Pulling into a newly created or previously unlinked EZ event layer auto-links that layer back to the MA3 source track coord.
- Batch pull planning can target `+ Create New Layer Per Source Track...` when multiple MA3 source tracks are selected.

## Development Tools

Located in `dev/`:

| File | Purpose |
|------|---------|
| `ma3_harness_cli.py` | Unified ping/version/health/browse/smoke CLI for the MA3 harness |
| `test.lua` | Test plugin for hooking into MA3 objects |
| `timecode_explorer.lua` | Systematic exploration of timecode structure |
| `timecode_helper.lua` | Standalone helper functions for timecode work |
| `TrackDump.txt` | Example dump output from MA3 |

## Documentation

### Core References

| Document | Description |
|----------|-------------|
| `MA3_LUA_API_REFERENCE.md` | Complete API reference, organized by category |
| `MA3_LEARNINGS.md` | Practical gotchas and working patterns |
| `MA3_SEQUENCE_MANAGEMENT.md` | Sequence helper contract and reply payloads for MA3 push prep |
| `MA3_EXPERT_STANDARD.md` | Standard source stack and live DataPool documentation workflow |
| `MA3_PLUGIN_DEVELOPMENT_GUIDE.md` | Full development walkthrough |
| `FIXTURE_PRESET_MENTAL_MODEL.md` | Operator-facing mental model for fixtures, preset modes, and fixture hierarchy |
| `docs/architecture/MA3-HARDWARE-VALIDATION-2026-05-03.md` | Current localhost hardware-validation evidence and remaining Wave 2 gaps |

### Technical Details

| Document | Description |
|----------|-------------|
| `ARCHITECTURE.md` | EchoZero <-> MA3 OSC architecture |
| `MA3_DATA_STRUCTURES.md` | MA3 object model documentation |
| `TIMECODE_STRUCTURE.md` | Timecode-specific hierarchy |
| `EVENT_PROPERTIES.md` | Event object properties |
| `MA3_CONNECTION_FLOW.md` | Communication protocol |
| `DataPool/` | Raw MA3 structure dumps |

## Installation

1. Copy the plugin bundle to MA3:
   ```
   [MA3 Show]/datapools/plugins/EZ/
   ```

2. Load in grandMA3: **Setup > Plugins**

3. Test connection:
   ```
   Lua "EZ.Ping()"
   ```

4. After editing plugin files, reload from disk with:
   ```
   RP
   ```

`RP` is the working reload command here. It is the short form of
`ReloadAllPlugins`.

## For AI Agents

When working on MA3 integrations:

1. **API Reference**: Start with `docs/MA3_LUA_API_REFERENCE.md`
   - Functions organized by category
   - [USED] markers show what EchoZero actively uses
   
2. **Learnings**: Check `docs/MA3_LEARNINGS.md` before coding
   - 12 critical discoveries
   - Common gotchas checklist
   - Patterns that work

3. **Deep Dive**: For new features, read:
   - `docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md` - Complete walkthrough
   - `docs/TIMECODE_STRUCTURE.md` - Hierarchy details
   - `docs/DataPool/` - Raw structure dumps

4. **Examples**: Study `plugins/echozero.lua`, `plugins/timecode.lua`, and the
   active live bundle under `gma3_library/datapools/plugins/EZ/`
   - working browse/query patterns
   - hook system implementation
   - OSC communication

## Testing

Listen for MA3 messages:
```bash
python MA3/dev/listen_for_ma3.py
```

Quick OSC test:
```bash
python MA3/dev/quick_osc_test.py
```

Integration test:
```bash
python MA3/dev/test_ma3_integration.py
```

Unified harness smoke:
```bash
python MA3/dev/ma3_harness_cli.py --json smoke
```

Harness result with native terminal feedback sidecar:
```bash
python MA3/dev/ma3_harness_cli.py --json --terminal-feedback analyze-cue-recipe-state --sequence-no 12 --cue-no 2
```

Structured DataPool browse for Codex/OpenClaw via our custom OSC/Lua layer:
```bash
python MA3/dev/ma3_harness_cli.py --json datapool-children --path Timecodes/1/2
python MA3/dev/ma3_harness_cli.py --json datapool-object --path Timecodes/1/2/3
```

These OSC calls are for EchoZero-facing custom functions we build on top of
MA3. They are not the raw MA console authority.

Use the MA terminal/CLI for native MA inspection such as `Dump()`,
`PropertyCount()`, `PropertyName()`, `PropertyType()`, `PropertyInfo()`, and
direct `List` / `ChangeDestination` output. In the live terminal, property
introspection is currently used as handle methods such as
`DataPool()[6][1]:PropertyCount()`.

Standard live DataPool documentation bundle:
```bash
python MA3/dev/ma3_datapool_documenter.py --output-dir artifacts/ma3-datapool/latest
```

This bundle maps the hierarchy exposed by our custom layer and generates a
terminal-first capture plan. Native MA attributes and dump evidence must come
from the MA terminal/CLI.

Standard native class/property catalog pass:
```bash
python MA3/dev/ma3_terminal_class_catalog.py \
  --sample-json artifacts/ma3-terminal-crawl/sample/datapool_sample_crawl.json \
  --output-dir artifacts/ma3-terminal-crawl/class-catalog/latest
```

This pass probes sampled MA objects through the live terminal and writes a
class catalog plus preset-focused notes.

Hardware validation report bundle:
```bash
python MA3/dev/ma3_harness_cli.py --json validation-report
```

Hardware validation report bundle with triggered receive proof:
```bash
python MA3/dev/ma3_harness_cli.py --json validation-report --receive-duration-seconds 2 --receive-trigger-command "EZ.Play(1)"
```

Receive-path capture:
```bash
python MA3/dev/ma3_harness_cli.py --json receive-capture --duration-seconds 5 --ping-first
```

Triggered receive-path capture:
```bash
python MA3/dev/ma3_harness_cli.py --json receive-capture --duration-seconds 2 --trigger-command "EZ.Play(1)"
```

Preset-reference phaser fixture set:
```bash
python MA3/dev/ma3_harness_cli.py --json create-phaser-fixture-set --selection-command "Fixture 1 Thru 4"
```

Plain-text object report for one preset and its child recipe objects:
```bash
python MA3/dev/ma3_harness_cli.py datapool-report --preset-ref 21.221 --depth 1
```

Explicit OSC preset pool list:
```bash
python MA3/dev/ma3_harness_cli.py --json list-presets --preset-type 21
```

Explicit OSC preset description:
```bash
python MA3/dev/ma3_harness_cli.py describe-preset --preset-ref 21.221
```

Explicit OSC preview for preset replacement filtered by group:
```bash
python MA3/dev/ma3_harness_cli.py --json preview-replace-preset-when-group \
  --preset-type 21 \
  --source-preset-ref "Preset 21.221" \
  --dest-preset-ref "Preset 21.222" \
  --group-filter "Drums" \
  --sequence-numbers "12"
```

Explicit OSC apply for preset replacement filtered by group:
```bash
python MA3/dev/ma3_harness_cli.py --json replace-preset-when-group \
  --preset-type 21 \
  --source-preset-ref "Preset 21.221" \
  --dest-preset-ref "Preset 21.222" \
  --group-filter "Drums" \
  --sequence-numbers "12"
```

Canonical app-path pull workspace smoke:
```bash
python MA3/dev/ma3_app_path_smoke.py --json
```

## Key Learnings Summary

| Issue | Solution |
|-------|----------|
| Property access returns nil | Use `:Children()` for hierarchy |
| Track 1 not working | Index 1 is "Marker", user tracks start at 2 |
| Time values wrong | Divide by 16777216 for large values |
| Hooks stop working | Callbacks must be global functions |
| Delete() fails | Call on parent: `subTrack:Delete(index)` |
| Writes don't stick | Use `:Set("Prop", value)` not assignment |

See `docs/MA3_LEARNINGS.md` for complete details.
