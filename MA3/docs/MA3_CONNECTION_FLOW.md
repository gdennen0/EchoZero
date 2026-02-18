# MA3 Connection Flow

Defines the connection lifecycle, state management, and when to call each function.

---

## Connection States

```
DISCONNECTED -> CONNECTING -> CONNECTED -> SYNCING -> READY
                    |             |           |
                    v             v           v
                 FAILED      STALE      SYNC_ERROR
```

### State Definitions

| State | Description | UI Indicator |
|-------|-------------|--------------|
| DISCONNECTED | No listener active | Gray dot |
| CONNECTING | Listener started, waiting for MA3 | Yellow dot, "Waiting for MA3..." |
| CONNECTED | Received ping/status from MA3 | Blue dot |
| SYNCING | Fetching track structure | Blue dot, "Syncing..." |
| READY | Tracks loaded, hooks active | Green dot |
| FAILED | Listener failed to start | Red dot, error message |
| STALE | No ping received > 30 seconds | Yellow dot, "Connection stale" |
| SYNC_ERROR | Error fetching structure | Orange dot, error details |

---

## Connection Lifecycle

### 1. Start Listening

**Trigger**: User clicks "Start Listening" or panel opens with saved port

**Actions**:
1. Create UDP socket on specified port
2. Set state to CONNECTING
3. Start ping timeout timer (30 seconds)
4. Wait for incoming messages

**Failure Handling**:
- Port in use: Show error, suggest different port
- Socket error: Show error, set state to FAILED

### 2. Receive First Message

**Trigger**: Any OSC message received from MA3

**Actions**:
1. Set state to CONNECTED
2. Update last_ping_time
3. Show "Connected to MA3"

### 3. Fetch Structure (On Demand)

**Trigger**: User clicks "Sync" or opens "Add MA3 Track" dialog

**Actions**:
1. Set state to SYNCING
2. Call `EZ.GetTrackGroups(tcNo)` for configured timecode
3. Wait for response
4. For each track group with tracks > 0, call `EZ.GetTracks(tcNo, tgNo)`
5. When all tracks received, set state to READY

**Important**: Do NOT automatically fetch on every panel open!

### 4. Monitor Connection

**Trigger**: Timer every 10 seconds

**Actions**:
1. Check last_ping_time
2. If > 30 seconds ago, set state to STALE
3. If > 60 seconds ago, set state to DISCONNECTED

### 5. Stop Listening

**Trigger**: User clicks "Stop", panel closes, or project closes

**Actions**:
1. Unhook all tracks: `EZ.UnhookAllTracks()`
2. Close UDP socket
3. Set state to DISCONNECTED

---

## When to Call Each Function

### EZ.ping()

**Call When**:
- Debugging connection issues
- Manual connection test

**Do NOT Call**:
- Automatically on timer (wastes resources)
- On every UI interaction

### EZ.GetTimecodes()

**Call When**:
- User requests timecode list
- Initial setup wizard

**Do NOT Call**:
- Automatically (timecodes rarely change)

### EZ.GetTrackGroups(tcNo)

**Call When**:
- User clicks "Sync" or "Refresh"
- User opens "Add MA3 Track" dialog AND no cached data exists
- Timecode selection changes

**Do NOT Call**:
- Every time panel opens (use cached data)
- On every incoming message
- Multiple times for same timecode without user action

### EZ.GetTracks(tcNo, tgNo)

**Call When**:
- After receiving trackgroups.list for a track group with track_count > 0
- ONCE per track group, not repeatedly

**Do NOT Call**:
- If tracks already cached for this track group
- Before getting track groups first

### EZ.HookTrack(tcNo, tgNo, trackNo)

**Call When**:
- User selects a track for sync
- User clicks "Start Sync" on a layer

**Do NOT Call**:
- For all tracks at once (hook only what's needed)
- If track already hooked

### EZ.UnhookTrack(tcNo, tgNo, trackNo)

**Call When**:
- User deselects a track
- Layer sync is stopped

**Do NOT Call**:
- Unnecessarily (check if actually hooked first)

### EZ.UnhookAllTracks()

**Call When**:
- Project closes
- Block is deleted
- User stops listening

---

## Caching Strategy

### What to Cache

| Data | Cache Key | Invalidation |
|------|-----------|--------------|
| Track groups | `tc_{tcNo}_trackgroups` | User clicks "Refresh" |
| Tracks | `tc_{tcNo}_tg_{tgNo}_tracks` | User clicks "Refresh" |
| Events | `tc_{tcNo}_tg_{tgNo}_tr_{trackNo}_events` | Hook change notification |

### Cache Implementation

```python
class MA3DataCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._max_age = 300  # 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        # Check age
        age = time.time() - self._timestamps.get(key, 0)
        if age > self._max_age:
            del self._cache[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
```

---

## Reducing Redundant Calls

### Current Problem

Looking at logs, `GetTracks` is called multiple times:
1. When panel opens
2. When dialog opens
3. When another message is received

### Solution

1. **Add cache check before fetching**:
```python
def _fetch_tracks_if_needed(self, tc: int, tg: int):
    cache_key = f"tc_{tc}_tg_{tg}_tracks"
    
    if self._cache.get(cache_key):
        self._log(f"Using cached tracks for TC{tc}.TG{tg}")
        return
    
    self._send_lua_command(f"EZ.GetTracks({tc}, {tg})")
```

2. **Remove auto-fetch from track groups handler**:
```python
def _handle_trackgroups_list_v2(self, message: OSCMessage):
    # Store track groups info but DON'T auto-fetch tracks
    # Let user trigger track fetch when needed
    pass
```

3. **Fetch tracks only when dialog opens**:
```python
def _on_add_ma3_track_clicked(self):
    # Only now fetch tracks if not cached
    for tg in self._ma3_track_groups.get(tc, []):
        self._fetch_tracks_if_needed(tc, tg['no'])
```

---

## Error States and Recovery

### Connection Errors

| Error | Cause | Recovery |
|-------|-------|----------|
| "Port in use" | Another app using port | Change port, restart listener |
| "Socket error" | System issue | Restart EchoZero |
| "No response" | MA3 not sending | Check MA3 plugin loaded, run EZ.ping() |
| "Parse error" | Malformed message | Check plugin version match |

### Sync Errors

| Error | Cause | Recovery |
|-------|-------|----------|
| "Timecode not found" | Wrong TC number | Check timecode exists in MA3 |
| "No track groups" | Empty timecode | Create track groups in MA3 |
| "Hook failed" | Unknown | Try again, check MA3 console |

### UI Error Display

```python
def _show_connection_error(self, error: str, recoverable: bool = True):
    self._status_label.setText(f"Error: {error}")
    self._status_label.setStyleSheet("color: red;")
    
    if recoverable:
        self._retry_btn.setVisible(True)
    else:
        self._log(f"FATAL: {error}")
        self._log("Please restart EchoZero")
```

---

## Hook State Machine

```
UNHOOKED -> HOOKING -> HOOKED -> RECEIVING -> HOOKED
               |                      |
               v                      v
           HOOK_FAILED           ERROR_RECOVERY
```

### Hook Tracking in EchoZero

```python
@dataclass
class HookedTrack:
    tc: int
    tg: int
    track: int
    hooked_at: float
    last_event_at: float = 0
    event_count: int = 0
    status: str = "hooked"

class HookManager:
    def __init__(self):
        self._hooked: Dict[str, HookedTrack] = {}
    
    def hook(self, tc: int, tg: int, track: int) -> bool:
        coord = f"tc{tc}_tg{tg}_tr{track}"
        if coord in self._hooked:
            return False  # Already hooked
        
        # Send hook command to MA3
        send_lua_command(f"EZ.HookTrack({tc}, {tg}, {track})")
        
        self._hooked[coord] = HookedTrack(
            tc=tc, tg=tg, track=track,
            hooked_at=time.time()
        )
        return True
    
    def unhook(self, tc: int, tg: int, track: int) -> bool:
        coord = f"tc{tc}_tg{tg}_tr{track}"
        if coord not in self._hooked:
            return False
        
        send_lua_command(f"EZ.UnhookTrack({tc}, {tg}, {track})")
        del self._hooked[coord]
        return True
    
    def unhook_all(self):
        send_lua_command("EZ.UnhookAllTracks()")
        self._hooked.clear()
    
    def is_hooked(self, tc: int, tg: int, track: int) -> bool:
        coord = f"tc{tc}_tg{tg}_tr{track}"
        return coord in self._hooked
```

---

## Summary: Recommended Flow

```
1. User opens ShowManager panel
   -> Start listener if not running
   -> Show CONNECTING state

2. First message received from MA3
   -> Set CONNECTED state
   -> Do NOT auto-fetch anything

3. User clicks "Add MA3 Track" 
   -> Check cache for track groups
   -> If empty: fetch GetTrackGroups(tc)
   -> Wait for response
   -> For each TG, check cache for tracks
   -> If empty: fetch GetTracks(tc, tg)
   -> Show dialog with available tracks

4. User selects track and clicks "Add"
   -> Create layer in Editor
   -> Hook the track: EZ.HookTrack(...)
   -> Show as "Hooked" in UI

5. Track change notification received
   -> Update events in Editor layer
   -> Show sync indicator

6. User closes panel / project
   -> Unhook all: EZ.UnhookAllTracks()
   -> Stop listener
```

---

*Last Updated: January 2026*
