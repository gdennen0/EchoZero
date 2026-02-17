# grandMA3 Integration Guide

EchoZero now supports bidirectional communication with grandMA3 lighting consoles via UDP.

## Architecture

### Components

1. **MA3 Lua Plugin** (`EchoZeroBridge.lua`)
   - Runs inside grandMA3
   - Hooks into MA3 objects using `HookObjectChange()`
   - Sends UDP messages to EchoZero when objects change

2. **EchoZero MA3 Communication Service**
   - Listens for UDP messages on configurable port (default: 9000)
   - Parses messages into structured events
   - Publishes `MA3MessageReceived` events to the event bus

### Message Flow

```
grandMA3 Object Change
    ↓
HookObjectChange() callback
    ↓
EchoZeroBridge.lua formats message
    ↓
UDP send to EchoZero (127.0.0.1:9000)
    ↓
MA3CommunicationService receives
    ↓
Parse message → MA3Message
    ↓
Publish MA3MessageReceived event
    ↓
Event subscribers can react
```

## Installation

### 1. Install MA3 Plugin

The MA3 plugin is located in the EchoZero repository:

```
EchoZero/ma3_plugins/EchoZeroBridge/EchoZeroBridge.lua
```

Copy it to your grandMA3 show directory:

```bash
cp ma3_plugins/EchoZeroBridge/EchoZeroBridge.lua [MA3 Show]/datapools/plugins/
```

In grandMA3:
- Go to **Setup > Plugins**
- Load `EchoZeroBridge`

### 2. Configure MA3 Plugin

In grandMA3 command line:

```
Lua "EchoZeroBridge.Configure()"
```

Enter:
- **EchoZero IP**: `127.0.0.1` (or your EchoZero machine's IP)
- **EchoZero Port**: `9000` (default)

### 3. Start Monitoring

```
Lua "EchoZeroBridge.StartHooks()"
```

This hooks into the Sequence Pool and begins sending change notifications.

## EchoZero Configuration

### Settings

MA3 communication is configured via application settings:

```python
from src.application.settings import AppSettingsManager

app_settings = services.app_settings

# Enable/disable listening
app_settings.ma3_listen_enabled = True

# Configure port and address
app_settings.ma3_listen_port = 9000
app_settings.ma3_listen_address = "127.0.0.1"
```

### Default Settings

- **Enabled**: `True`
- **Port**: `9000`
- **Address**: `127.0.0.1` (localhost)

Settings are persisted automatically and loaded on startup.

## Using MA3 Events

### Subscribe to Events

```python
from src.application.events import MA3MessageReceived, EventBus

def handle_ma3_sequence_change(event: MA3MessageReceived):
    """Handle sequence changes from MA3"""
    data = event.data
    object_type = data['object_type']  # "sequence"
    object_name = data['object_name']   # "Song1"
    change_type = data['change_type']   # "changed"
    timestamp = data['timestamp']
    ma3_data = data['ma3_data']         # Additional data
    
    print(f"MA3 Sequence '{object_name}' {change_type} at {timestamp}")

# Subscribe
event_bus.subscribe(MA3MessageReceived, handle_ma3_sequence_change)
```

### Access MA3 Communication Service

```python
from src.application.services.ma3_communication_service import MA3CommunicationService

# Get service from services container
ma3_service = services.ma3_communication_service

# Check if listening
if ma3_service.is_listening():
    print("Listening for MA3 messages")

# Register custom handler for specific message types
def handle_sequence_changes(message: MA3Message):
    print(f"Sequence {message.object_name} changed")

ma3_service.register_handler("sequence", "changed", handle_sequence_changes)

# Send message back to MA3 (future: bidirectional)
ma3_service.send_message("Hello from EchoZero", "127.0.0.1", 9001)
```

## Message Format

Messages from MA3 use a simple pipe-delimited format:

```
type=sequence|name=Song1|change=changed|timestamp=1234567890|no=1
```

### Parsed Structure

```python
MA3Message(
    object_type="sequence",
    object_name="Song1",
    change_type="changed",
    timestamp=1234567890.0,
    data={"no": "1"}
)
```

## Testing

### Test from MA3

Send a test message from grandMA3:

```
Lua "EchoZeroBridge.TestConnection()"
```

This sends a test message that EchoZero should receive and log.

### Test from EchoZero

You can manually test the UDP receiver:

```python
import socket

# Send test message
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = "type=test|name=EchoZeroBridge|change=test_message|timestamp=1234567890"
sock.sendto(message.encode('utf-8'), ('127.0.0.1', 9000))
sock.close()
```

Check EchoZero logs for the received message.

## Extending

### Adding More Object Hooks

Edit `EchoZeroBridge.lua` to hook into additional MA3 objects:

```lua
-- Hook into cue pool
local cuePool = DataPool().Cues
local cueHookId = HookObjectChange(on_cue_change, cuePool, pluginHandle)
hooks.cue_pool = cueHookId

-- Hook into executor pool
local execPool = DataPool().Executors
local execHookId = HookObjectChange(on_executor_change, execPool, pluginHandle)
hooks.executor_pool = execHookId
```

### Custom Message Handlers

Register handlers for specific object/change combinations:

```python
def handle_cue_created(message: MA3Message):
    # React to cue creation
    pass

ma3_service.register_handler("cue", "created", handle_cue_created)
```

## Troubleshooting

### Messages Not Received

1. **Check EchoZero is running**: Service must be active
2. **Verify settings**: Check `ma3_listen_enabled`, port, address
3. **Check firewall**: UDP port must be open
4. **Test connection**: Use `EchoZeroBridge.TestConnection()`
5. **Check logs**: Look for MA3CommunicationService messages

### Hook Not Working

1. **Verify plugin loaded**: Check MA3 plugin list
2. **Check hook started**: Look for hook ID in MA3 command line
3. **Verify object exists**: Ensure the pool you're hooking exists
4. **Restart plugin**: Unload and reload, then restart hooks

## Future Enhancements

- [ ] OSC protocol support (in addition to UDP)
- [ ] Bidirectional communication (EchoZero → MA3 commands)
- [ ] Additional object types (cues, executors, timecode, etc.)
- [ ] Message filtering/selection
- [ ] Automatic hook management
- [ ] Sequence/cue synchronization
- [ ] Timecode sync

## Testing

### Quick Test

See `ma3_plugins/EchoZeroBridge/QUICK_TEST.md` for step-by-step testing instructions.

### Test Scripts

1. **Standalone UDP test** (`test_2way.py`):
   - Tests raw UDP communication
   - Doesn't require EchoZero to be running
   - Good for verifying network connectivity

2. **EchoZero service test** (`test_from_echozero.py`):
   - Uses EchoZero's MA3CommunicationService
   - Requires EchoZero to be running
   - Tests full integration

## Files

- **MA3 Plugin**: `ma3_plugins/EchoZeroBridge/EchoZeroBridge.lua`
- **Test Scripts**: `ma3_plugins/EchoZeroBridge/test_*.py`
- **EchoZero Service**: `src/application/services/ma3_communication_service.py`
- **Event**: `src/application/events/events.py` (MA3MessageReceived)
- **Settings**: `src/application/settings/app_settings.py` (MA3 settings)

