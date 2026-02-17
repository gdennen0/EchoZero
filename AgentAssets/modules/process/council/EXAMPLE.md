# Example Council Decision: Adding Real-Time Audio Preview

**This is a hypothetical example showing how the council would analyze a real proposal.**

---

## Proposal

**Title:** Add Real-Time Audio Preview Feature

**Problem:**
Users want to hear audio at different stages of their processing pipeline without having to execute the entire graph and export to files.

**Proposed Solution:**
Add a "Preview" button to blocks that:
- Plays audio from that block's output in real-time
- Shows waveform visualization
- Provides playback controls (play, pause, stop, seek)
- Works for any block with audio output

**Implementation Plan:**
1. Add Qt audio playback system
2. Integrate QMediaPlayer or similar
3. Add preview UI to each audio block
4. Cache audio data in memory for playback
5. Add waveform visualization widget
6. Implement playback controls

**Estimated Effort:** 2-3 weeks, ~1500 LOC

**New Dependencies:**
- Qt Multimedia (QMediaPlayer)
- Additional visualization library

---

## Council Analysis

### Architect Analysis

**Structural Concerns:**

**Positive:**
- Preview is a valid use case for audio applications
- Could fit naturally into block lifecycle

**Concerns:**
1. **Layer violation:** Preview UI shouldn't be in block entities (domain layer)
2. **Coupling:** Blocks would become coupled to Qt/audio playback
3. **Abstraction:** No abstraction between domain and playback mechanism
4. **Scope:** "Preview UI to each audio block" suggests deep integration

**Alternative Structural Approach:**
- Keep blocks pure (no UI knowledge)
- Add PreviewService in application layer
- GUI requests preview via ApplicationFacade
- Service handles playback independently
- Blocks just provide data, don't know about playback

**Impact on Architecture:**
- As proposed: Violates layer boundaries, couples domain to Qt
- With alternative: Clean separation, proper layering

**Vote: Reject with Alternative**

**Reasoning:**
The proposed implementation would embed UI concerns into the domain/application layers. This breaks our layered architecture and couples blocks to playback technology.

**Alternative:**
```python
# In ApplicationFacade
def preview_block_output(self, block_id: str, output_port: str) -> CommandResult:
    """Get audio data for preview - GUI handles playback"""
    audio_data = self.get_block_output_data(block_id, output_port)
    return CommandResult.success(data=audio_data)

# In GUI layer only
class AudioPreviewWidget(QWidget):
    def __init__(self, facade):
        self.facade = facade
        self.media_player = QMediaPlayer()
    
    def preview_block(self, block_id):
        result = self.facade.preview_block_output(block_id, "audio")
        if result.success:
            self.play_audio(result.data)
```

This keeps domain clean, UI in presentation layer, proper separation.

---

### Systems Analysis

**Infrastructure Concerns:**

**Positive:**
- Real-time preview is technically feasible
- Qt Multimedia is stable and well-tested

**Concerns:**
1. **Resource Management:** Multiple blocks could have preview players open
2. **Memory:** Caching audio data for all previewable blocks could be expensive
3. **Threading:** Audio playback typically needs separate thread
4. **Lifecycle:** Who cleans up players? When?
5. **Dependencies:** Qt Multimedia adds significant dependency
6. **Failure Modes:** Audio device busy, format incompatible, playback errors

**Resource Analysis:**
- Audio data in memory: ~10MB per minute per block
- Multiple previews: Could accumulate quickly
- QMediaPlayer instances: Need proper lifecycle management
- We just fixed memory leaks - this could reintroduce them

**Performance Impact:**
- Loading audio: Minimal (already loaded)
- Playback: Separate thread, minimal impact
- Visualization: Could be expensive if not optimized

**Failure Modes:**
- Audio device unavailable
- Unsupported format
- Playback crashes
- Memory exhaustion with large files

**Vote: Approve with Conditions**

**Reasoning:**
Technically feasible but resource management is critical. We just fixed memory leaks - must not reintroduce them.

**Conditions:**
1. Implement proper cleanup (learned from recent memory leak fixes)
2. Limit to one preview at a time initially
3. Clear memory management strategy for cached audio
4. Comprehensive error handling for all failure modes
5. Preview service must be self-contained in GUI layer

---

### UX Analysis

**User Experience Concerns:**

**Positive:**
- Preview is valuable for users
- Immediate feedback improves workflow
- Reduces execute → export → open cycle

**Concerns:**
1. **Complexity:** Full preview UI on every audio block might be cluttered
2. **Discoverability:** Where is preview UI? Always visible?
3. **Expectations:** Users might expect real-time effects (we're offline processing)
4. **Workflow:** How does this fit into current CLI workflow? (We don't have GUI yet!)
5. **Consistency:** Preview only on audio blocks? What about events?

**Workflow Analysis:**

Current workflow:
```
1. Build graph
2. Execute
3. ExportAudio to file
4. Open in external player
5. Hear result
6. Adjust parameters
7. Repeat
```

With preview (if we had GUI):
```
1. Build graph
2. Execute (or execute up to here?)
3. Click preview
4. Hear result
5. Adjust parameters
6. Repeat
```

**But we don't have a GUI yet!**

This feature assumes GUI exists. We're CLI-only currently.

**Learning Curve:**
- Where to click preview?
- Which output port gets previewed?
- How to stop/control playback?
- What if block hasn't executed yet?

**Vote: Reject with Alternative**

**Reasoning:**
We don't have a GUI yet! This feature is GUI-dependent but proposal doesn't address that. Also, trying to add preview UI to every block is cluttered.

**Alternative:**
1. Wait until GUI development starts
2. Then build preview as dedicated panel, not per-block
3. Select block → preview panel shows its outputs
4. Cleaner, more flexible, better UX
5. Allows comparison of multiple blocks

Or, even simpler for CLI:
```
EZ> preview LoadAudio1 audio
[plays audio]
```

Simple CLI command that plays audio. No UI needed. Could implement now.

---

### Pragmatic Analysis

**Implementation Concerns:**

**Positive:**
- Qt Multimedia is well-documented
- Audio playback is solved problem
- Could deliver value quickly

**Concerns:**
1. **Scope:** 1500 LOC is substantial
2. **Dependencies:** Qt Multimedia is large dependency
3. **Testing:** How do we test audio playback?
4. **GUI Assumption:** We don't have GUI yet!
5. **Priority:** Is this more important than getting basic GUI working?
6. **Maintenance:** Another system to maintain

**Complexity Analysis:**
- Audio playback: Medium complexity
- Waveform visualization: Medium-High complexity
- Integration with blocks: High complexity (as proposed)
- UI for controls: Medium complexity
- Total: HIGH complexity

**MVP Analysis:**

What's the minimum?
- CLI command: `preview <block_id> <port>` plays audio
- Uses system audio player
- 50 lines of code
- No Qt dependency
- Works today (we have CLI)

**Testing:**
- Audio playback: Hard to test (requires audio device)
- UI interactions: Requires GUI testing framework
- Integration: Complex setup
- Overall: Testing burden is significant

**Maintenance:**
- Qt Multimedia updates
- Cross-platform audio device issues
- Memory management (as Systems noted)
- UI maintenance
- Overall: Non-trivial burden

**Vote: Reject with Alternative**

**Reasoning:**
Scope is too large, assumes GUI exists, and we can achieve 80% of value with 5% of complexity.

**Alternative (Immediate Value):**
```python
# Add to ApplicationFacade
def export_block_output_temp(self, block_id: str, output_port: str) -> CommandResult:
    """Export block output to temp file and open in system player"""
    audio_data = self.get_block_output_data(block_id, output_port)
    temp_file = self.export_to_temp(audio_data)
    os.system(f"open {temp_file}")  # macOS, use xdg-open on Linux
    return CommandResult.success(f"Playing {temp_file}")

# CLI
EZ> preview LoadAudio1 audio
Playing /tmp/echozero_preview_xyz.wav
```

**Result:**
- 20 lines of code
- No dependencies
- Works today
- Solves user problem
- Can enhance later when GUI exists

---

## Council Discussion

**Common Ground:**
- All members agree preview is valuable
- All members see issues with proposed approach
- All members prefer simpler solution

**Key Disagreements:**
- None - all members converge on simpler approach

**Alternative Explored:**

**Immediate (CLI-based):**
```
Command: preview <block_id> <output_port>
Implementation: Export to temp file, open in system player
Effort: Few hours, ~20 LOC
Value: 80% of user need met
Dependencies: None (use system player)
```

**Future (GUI-based):**
When GUI exists:
```
Preview Panel (not per-block):
- Select block in graph
- Panel shows all output ports
- Click to preview any output
- Waveform visualization
- Playback controls
- Clean, dedicated UI
```

**Consensus Reached:**
All members agree simpler approach is better.

---

## Unanimous Recommendation

**RECOMMENDATION: Use Alternative Approach (CLI Preview Command)**

The Council recognizes the value of audio preview but recommends a much simpler implementation that:
1. Works today (CLI exists, GUI doesn't)
2. Requires minimal code (~20 LOC vs 1500 LOC)
3. Adds no dependencies
4. Solves 80% of user need
5. Can be enhanced later

### Why Original Proposal Doesn't Work

**Architect Perspective:**
- Violates layer boundaries
- Couples domain to Qt
- Wrong abstraction level

**Systems Perspective:**
- Resource management concerns
- Memory leak risk
- Complex lifecycle management

**UX Perspective:**
- Assumes GUI exists (it doesn't)
- Cluttered if every block has preview UI
- Better as dedicated panel

**Pragmatic Perspective:**
- Too much scope (1500 LOC)
- Hard to test
- High maintenance burden
- We can get 80% value for 5% effort

### Recommended Implementation

**Phase 1: CLI Preview (Now)**

Add to ApplicationFacade:
```python
def preview_block_output(
    self, 
    block_id: str, 
    output_port: str = "audio"
) -> CommandResult:
    """
    Preview audio output from a block.
    Exports to temp file and opens in system audio player.
    """
    # Validate block exists and has been executed
    block = self.block_service.get_block(block_id)
    if not block:
        return CommandResult.error(f"Block {block_id} not found")
    
    # Get output data
    data_item = self.data_service.get_block_output(block_id, output_port)
    if not data_item or not isinstance(data_item, AudioDataItem):
        return CommandResult.error(f"No audio data at {block_id}.{output_port}")
    
    # Export to temp file
    import tempfile
    temp_path = tempfile.mktemp(suffix=".wav")
    data_item.save_audio(temp_path)
    
    # Open in system player
    import subprocess
    import platform
    
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", temp_path])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", temp_path])
    elif platform.system() == "Windows":
        subprocess.run(["start", temp_path], shell=True)
    
    return CommandResult.success(
        f"Playing {block.name}.{output_port} in system player",
        data={"temp_file": temp_path}
    )
```

CLI command:
```python
# In CommandParser
def _handle_preview(self, args, kwargs):
    block_id = args[0] if args else None
    output_port = args[1] if len(args) > 1 else "audio"
    return self.facade.preview_block_output(block_id, output_port)
```

**Effort:** 2-4 hours
**LOC:** ~30 (including error handling)
**Dependencies:** None (uses stdlib + system player)
**Risk:** Very low
**Value:** High (solves immediate need)

**Phase 2: GUI Preview Panel (When GUI Exists)**

When GUI development starts:
- Dedicated preview panel in GUI
- Select block → see all outputs
- Click to preview any output
- Built-in player with controls
- Waveform visualization
- Clean, professional UI

Can leverage Phase 1 facade method, just with GUI player instead of system player.

### Success Criteria

**Phase 1:**
- Users can preview audio outputs from CLI
- Opens in familiar system player
- Works cross-platform
- No new dependencies
- No resource leaks

**Phase 2 (Future):**
- Clean, intuitive preview UI
- Integrated into workflow
- Professional playback controls
- Visual feedback (waveform)
- Proper resource management

### Action Items

- [ ] Implement `preview_block_output` in ApplicationFacade
- [ ] Add `preview` CLI command
- [ ] Test on macOS, Linux, Windows
- [ ] Document in HANDBOOK.md
- [ ] Add to command help
- [ ] Consider adding tests (mock subprocess calls)

### Why This Is Better

**Simplicity:**
- 30 LOC instead of 1500 LOC (98% reduction)
- No new dependencies
- Uses existing system capabilities

**Immediate Value:**
- Works today with CLI
- Solves user problem now
- No waiting for GUI

**Future-Friendly:**
- When GUI exists, can build proper preview panel
- Facade method reusable
- Not locked into any particular UI approach

**Low Risk:**
- No resource management complexity
- No memory leaks
- No Qt coupling
- Easy to test and maintain

**Alignment with Core Values:**
- "Best part is no part" - leverages system player instead of building one
- "Simplicity and refinement" - solves problem simply
- Pragmatic delivery - quick win now, polish later

---

## Lessons from This Decision

**What Went Right:**
- All members identified fundamental issues
- Alternative emerged naturally
- Found simpler solution with more value
- Stayed aligned with core values

**What This Demonstrates:**
- Question assumptions (GUI doesn't exist yet!)
- Seek the 20% that gives 80% value
- Use existing tools (system player) before building
- Deliver value quickly, polish later
- Layer boundaries matter

**Key Takeaway:**
The best solution isn't always the most comprehensive. Often, it's the simplest thing that solves the actual user problem.

**The best part is no part. In this case, the best audio player is the one the user already has.**

---

*This example demonstrates the council process in action. Real decisions will vary but should follow this level of rigor and focus on simplicity.*

