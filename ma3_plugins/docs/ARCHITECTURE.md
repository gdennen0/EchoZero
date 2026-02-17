# EchoZero <-> MA3 OSC Architecture

This document describes the architecture for bidirectional OSC communication between EchoZero and grandMA3.

## Core Principle

**EchoZero is the brain, MA3 is the executor.**

- EchoZero does all heavy lifting: audio analysis, event detection, data processing
- EchoZero sends commands to MA3 via OSC
- MA3 receives commands and executes them on the lighting console
- MA3 sends acknowledgments and status back to EchoZero

## Communication Flow

```
+----------------+       OSC Commands         +----------------+
|                |  ----------------------->  |                |
|   EchoZero     |     /echozero/*            |    grandMA3    |
|  (Python)      |                            |     (Lua)      |
|                |  <-----------------------  |                |
+----------------+       OSC Responses        +----------------+
                         /ma3/*
```

### Ports

| Direction | Source | Destination | Port |
|-----------|--------|-------------|------|
| Commands  | EchoZero | MA3 | 9001 |
| Responses | MA3 | EchoZero | 9000 |

## OSC Address Scheme

### Commands (EchoZero -> MA3): `/echozero/*`

| Address | Args | Description |
|---------|------|-------------|
| `/echozero/ping` | - | Connection test |
| `/echozero/echo` | s:message | Echo test |
| `/echozero/status` | - | Request status |
| `/echozero/create_track_group` | i:tc_no, s:name | Create track group |
| `/echozero/create_track` | i:tc_no, i:tg_idx, s:name | Create track |
| `/echozero/create_event` | i:tc_no, i:tg_idx, i:track_idx, f:time_secs, s:type, [s:props] | Create event |
| `/echozero/clear_timecode` | i:tc_no | Clear all events |
| `/echozero/batch_start` | s:batch_id, i:count | Start batch operation |
| `/echozero/batch_end` | s:batch_id | End batch operation |

### Responses (MA3 -> EchoZero): `/ma3/*`

| Address | Args | Description |
|---------|------|-------------|
| `/ma3/pong` | i:timestamp | Ping response |
| `/ma3/echo` | s:message | Echo response |
| `/ma3/status` | s:state, i:port, s:info | Status response |
| `/ma3/ack` | s:command, ... | Command acknowledgment |
| `/ma3/error` | s:type, s:details | Error response |

## Architecture Components

### EchoZero Side (Python)

```
src/application/services/
    osc_bridge_service.py     # Main OSC service
        - OSCConfig           # Configuration dataclass
        - OSCMessage          # Parsed message dataclass
        - OSCBridgeService    # Send/receive OSC messages
        - EventExporter       # High-level event export
```

### MA3 Side (Lua)

```
ma3_plugins/EchoZeroBridge/
    osc_receiver.lua          # OSC command receiver
        - OSCReceiver.Start()     # Start listening
        - OSCReceiver.Stop()      # Stop listening
        - OSCReceiver.Check()     # Poll for messages
        - OSCReceiver.Configure() # Configure ports
    
ma3_plugins/timecode_helpers/
    manipulation.lua          # Low-level timecode manipulation
        - CreateEvent()           # Create timecode event
        - SetEventTime()          # Modify event time
        - GetTrackGroups()        # List track groups
        - etc.
```

## MA3 Command Spine

The MA3 "command spine" is a clean dispatch system for handling EchoZero commands.

### Design Principles

1. **Single Entry Point**: All commands go through `osc_receiver.lua`
2. **Command Registry**: Handlers registered by OSC address
3. **Standard Response Format**: All handlers send acknowledgment
4. **Error Isolation**: Each handler wrapped in pcall
5. **Logging**: Debug logging for troubleshooting

### Command Flow

```
1. OSC Message Received
        |
2. Parse OSC (address + args)
        |
3. Lookup Handler in registry
        |
4. Execute Handler (pcall wrapped)
        |
5. Handler calls timecode_helpers
        |
6. Send Response (/ma3/ack or /ma3/error)
```

### Adding New Commands

1. **Define OSC address**: `/echozero/your_command`
2. **Add handler in osc_receiver.lua**:
```lua
commands["/echozero/your_command"] = function(args)
    local param1 = args[1]
    local param2 = args[2]
    
    -- Do work using timecode_helpers
    local success = YourHelperFunction(param1, param2)
    
    -- Send response
    send_response("/ma3/ack", {"your_command", param1, success and "ok" or "failed"})
    return success
end
```
3. **Add Python method in OSCBridgeService**:
```python
def your_command(self, param1: int, param2: str) -> bool:
    return self.send("/echozero/your_command", param1, param2)
```

## Data Model Mapping

### EchoZero EventDataItem -> MA3 Timecode

```
EchoZero                          MA3
--------                          ---
EventDataItem                     Timecode
    |                                 |
    +-- events[]                      +-- TrackGroup
        |                                 |
        +-- Event                         +-- Track
            time: float                       |
            classification: str               +-- TimeRange
            duration: float                       |
            metadata: dict                        +-- CmdSubTrack
                                                      |
                                                      +-- CmdEvent
                                                          TIME: int (internal units)
                                                          TOKEN: command
```

### Time Conversion

MA3 uses internal time units: `16777216 units = 1 second`

```python
def seconds_to_ma3(seconds: float) -> int:
    return int(seconds * 16777216)

def ma3_to_seconds(ma3_time: int) -> float:
    return ma3_time / 16777216
```

## Event Export Workflow

```
1. User triggers export in EchoZero UI
        |
2. EventExporter.export_events() called
        |
3. Group events by classification
        |
4. Send batch_start
        |
5. For each classification:
        |
        +-- create_track(classification_name)
        |
        +-- For each event:
                |
                +-- create_event(time, type)
        |
6. Send batch_end
        |
7. Report results to UI
```

## Testing Strategy

### Level 1: Local Mock Test
- `test_osc_local.py` runs mock MA3 responder
- Tests OSCBridgeService without MA3
- Use for development and CI

### Level 2: OSC 2-Way Test
- `test_osc_2way.py` tests with real MA3
- Requires MA3 running with OSCReceiver
- Tests full communication path

### Level 3: Integration Test
- Test full EventDataItem -> MA3 export
- Verify events appear in MA3 timecode
- Manual verification of lighting timing

## Configuration

### EchoZero (Python)

```python
from src.application.services.osc_bridge_service import OSCBridgeService, OSCConfig

config = OSCConfig(
    listen_port=9000,      # EchoZero listens here
    ma3_port=9001,         # MA3 listens here
    ma3_address="127.0.0.1"
)

bridge = OSCBridgeService(config)
bridge.start_listening()
```

### MA3 (Lua)

```lua
-- Load plugin
Lua "dofile('/path/to/osc_receiver.lua')"

-- Configure
Lua "OSCReceiver.Configure()"

-- Start
Lua "OSCReceiver.Start()"

-- Check for messages (call periodically)
Lua "OSCReceiver.Check()"
```

## Future Enhancements

### Phase 2: Advanced Commands
- [ ] `delete_event` - Remove specific event
- [ ] `update_event` - Modify existing event
- [ ] `query_timecode` - Get timecode structure
- [ ] `sync_events` - Two-way sync

### Phase 3: Real-time Features
- [ ] Transport sync (timecode position)
- [ ] Live event triggering
- [ ] Beat/tempo synchronization

### Phase 4: UI Integration
- [ ] MA3 connection status in EchoZero
- [ ] Export progress visualization
- [ ] MA3 timecode browser

## Troubleshooting

### No Response from MA3

1. Check MA3 is running
2. Verify OSCReceiver is loaded: `Lua "print(OSCReceiver ~= nil)"`
3. Verify listening: `Lua "print(OSCReceiver.IsListening())"`
4. Check ports in firewall
5. Run `OSCReceiver.Check()` to poll for messages

### Parse Errors

1. Enable debug: `Lua "OSCReceiver.Debug(true)"`
2. Check message format in EchoZero logs
3. Verify type tags match expected arguments

### Event Creation Fails

1. Verify timecode exists
2. Check track group index (starts at 1)
3. Check track index (starts at 1, skip 0 which is Marker)
4. Verify timecode_helpers are loaded
