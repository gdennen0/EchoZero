# Current Demucs Implementation in EchoZero

## Architecture Overview

EchoZero integrates Demucs through a **thin CLI wrapper** approach:

```
User Command
    │
    ▼
ApplicationFacade
    │
    ▼
SeparatorBlock (Entity)
    │
    ▼
SeparatorBlockProcessor
    │
    ├─> Validate inputs (audio file exists)
    ├─> Build Demucs CLI command
    ├─> Execute subprocess (streaming output)
    ├─> Collect output stems
    └─> Return AudioDataItem list
```

### Why CLI Wrapper?

**Advantages:**
1. **Simple Integration:** No deep coupling with Demucs internals
2. **Demucs Handles Complexity:** Model downloads, device detection, updates
3. **Clear Separation:** Demucs is external tool, not embedded library
4. **Maintainable:** Demucs CLI changes are less breaking than library API
5. **Debuggable:** User can run same command manually
6. **Observable:** Real-time output streaming, progress visible

**Disadvantages (Accepted Trade-offs):**
1. **Subprocess Overhead:** ~1 second startup (negligible vs minutes of processing)
2. **File I/O:** Must write/read files (acceptable for offline processing)
3. **Less Control:** Limited to CLI options (sufficient for needs)

**Decision:** CLI wrapper aligns with "simplicity" - let Demucs be Demucs.

## Component Breakdown

### 1. SeparatorBlock Entity

**Location:** `src/domain/entities/block.py` (base class)

**Type:** `"Separator"`

**Ports:**
- Input: `audio` (Audio type)
- Output: `audio` (Audio type, returns list of AudioDataItem)

**Metadata Fields:**
```python
{
    "model": "htdemucs",           # Model name
    "device": "auto",              # cpu, cuda, mps, auto
    "output_dir": "/path/to/stems", # Optional custom directory
    "output_format": "wav",        # wav or mp3
    "mp3_bitrate": "320",          # 128, 192, 320
    "two_stems": None,             # None, "vocals", "drums", "bass", "other"
    "separator_settings": {        # Alternative nested config
        "model": "htdemucs"
    }
}
```

**Notes:**
- Metadata is flexible (dict)
- Defaults in processor, not entity
- User can set via commands or project file

### 2. SeparatorBlockProcessor

**Location:** `src/application/blocks/separator_block.py` (323 lines)

**Responsibilities:**
1. Validate Demucs CLI availability
2. Detect best device (auto mode)
3. Build Demucs command with optimizations
4. Execute subprocess with streaming output
5. Handle errors (SSL, missing files, Demucs failures)
6. Collect output stems
7. Create AudioDataItem for each stem

**Key Methods:**

```python
def can_process(block: Block) -> bool:
    return block.type == "Separator"

def process(block, inputs, metadata) -> Dict[str, DataItem]:
    # 1. Validate inputs
    # 2. Check Demucs installed
    # 3. Get settings from metadata
    # 4. Auto-detect device if needed
    # 5. Build command
    # 6. Execute with streaming
    # 7. Collect stems
    # 8. Return AudioDataItem list
```

**Error Handling:**
- **Demucs not found:** Clear install instructions
- **SSL certificate errors:** Detailed fix steps (macOS Install Certificates.command)
- **File not found:** Check input path
- **Demucs failure:** Show full output, common issues
- **No stems produced:** Validate input file

### 3. Device Auto-Detection

**Function:** `detect_best_device()`

**Logic:**
```python
def detect_best_device() -> str:
    # 1. Check CUDA (NVIDIA GPU) - fully supported
    if torch.cuda.is_available():
        return "cuda"
    
    # 2. Fall back to CPU
    # Note: MPS (Apple Silicon) not used - FFT operations unsupported
    return "cpu"
```

**Why Not MPS?**
- PyTorch MPS backend doesn't support FFT operations (aten::_fft_r2c)
- Demucs requires FFT for spectral analysis
- Known PyTorch limitation: https://github.com/pytorch/pytorch/issues/77764
- Decision: Fall back to CPU on Apple Silicon (still functional, just slower)

**Future:** If PyTorch adds MPS FFT support, update auto-detection.

### 4. Command Building

**Example Command:**
```bash
demucs -n htdemucs_ft \
       -o /output/directory \
       -d cuda \
       --two-stems vocals \
       --mp3 \
       --mp3-bitrate 320 \
       /path/to/input.wav
```

**Flags:**
- `-n MODEL`: Model name (not --model)
- `-o DIR`: Output directory
- `-d DEVICE`: Device (cpu, cuda)
- `--two-stems STEM`: Fast 2-stem mode
- `--mp3`: Output MP3 instead of WAV
- `--mp3-bitrate N`: MP3 quality

**Notes:**
- Model downloads happen automatically (Demucs handles this)
- Cache location: `~/.cache/torch/hub/checkpoints/`
- User never needs to manually download models

### 5. Output Handling

**Demucs Output Structure:**
```
output_dir/
└── model_name/           # e.g., htdemucs_ft
    └── track_name/       # Input filename without extension
        ├── vocals.wav
        ├── drums.wav
        ├── bass.wav
        └── other.wav
```

**Processor Logic:**
1. Scan output_dir for model directories
2. Find track directories
3. Collect all .wav or .mp3 files
4. Create AudioDataItem for each stem
5. Return as list on "audio" port

**DataItem Structure:**
```python
AudioDataItem(
    id="",                    # Generated
    block_id=block.id,
    name="SeparatorBlock1_vocals",
    type="Audio",
    created_at=datetime.utcnow(),
    file_path="/path/to/output/htdemucs_ft/track/vocals.wav"
)
```

### 6. Block Registry Entry

**Location:** `src/application/block_registry.py` (lines 427-451)

**Metadata:**
```python
BlockTypeMetadata(
    name="Demucs Separator",
    type_id="Separator",
    description="Separate audio into stems using Demucs (outputs multiple stems on single port)",
    category="Processing",
    inputs={"audio": AUDIO_TYPE},
    outputs={"audio": AUDIO_TYPE},  # List of AudioDataItem
    tags=["separator", "demucs", "stem", "audio"],
    metadata_schema={...},  # set_model command definition
    commands=[
        {
            "name": "set_model",
            "usage": "set_model <demucs_model>",
            "description": "Choose which Demucs model to run",
            "params": [...]
        },
        {
            "name": "list_models",
            "description": "List available Demucs models with descriptions"
        }
    ]
)
```

## User Workflows

### Basic Separation

```
# Create blocks
add_block LoadAudio loader
add_block Separator sep

# Connect
connect loader audio sep audio

# Configure model (optional, defaults to htdemucs)
sep set_model htdemucs_ft

# Execute
execute

# Results: sep produces multiple AudioDataItem on "audio" port
# Each stem (vocals, drums, bass, other) is a separate AudioDataItem
```

### Advanced Workflow (Note Extraction per Stem)

```
# Load and separate
add_block LoadAudio loader
add_block Separator sep
connect loader audio sep audio

# Extract notes from vocals
add_block NoteExtractorBasicPitch notes
connect sep audio notes audio

# Visualize
add_block PlotEvents plot
connect notes events plot events

# Execute
execute

# Notes: NoteExtractor receives list of stems, processes first by default
# Future enhancement: Allow stem selection
```

### Performance Optimization

```
# Fast 2-stem vocal isolation
add_block Separator sep
sep set_metadata two_stems vocals
sep set_metadata output_format mp3
sep set_metadata mp3_bitrate 192

# Results: 2x faster, smaller files
```

## Integration Points

### Application Layer

**ApplicationFacade:**
- `add_block("Separator")` - Creates SeparatorBlock
- `set_block_metadata(block_name, "model", "htdemucs_ft")` - Configure model
- `execute_project()` - Triggers processing

**BlockService:**
- Validates "Separator" block type exists
- Checks model name against DEMUCS_MODELS
- Provides model info via get_demucs_models_info()

**ExecutionEngine:**
- Topologically sorts (Separator after LoadAudio)
- Passes AudioDataItem from LoadAudio to Separator
- Receives list of AudioDataItems from Separator
- Continues execution with stems

### Domain Layer

**Block Entity:**
- Stores metadata (model, device, etc.)
- No Demucs-specific logic
- Generic enough for any processor type

**AudioDataItem:**
- Represents each stem
- file_path points to Demucs output
- name includes stem type (e.g., "sep_vocals")

### Infrastructure Layer

**BlockTypeRegistry:**
- Defines Separator block metadata
- Lists available models
- Command definitions

**Database:**
- Stores SeparatorBlock like any block
- Metadata stored as JSON
- AudioDataItem references stored

## Configuration Flexibility

### Setting Model: 3 Ways

**1. Via Command (Recommended):**
```
sep set_model htdemucs_ft
```

**2. Via Metadata Command:**
```
sep set_metadata model htdemucs_ft
```

**3. Via Project File:**
```json
{
  "blocks": [
    {
      "name": "sep",
      "type": "Separator",
      "metadata": {
        "model": "htdemucs_ft"
      }
    }
  ]
}
```

### Default Behavior

**If nothing configured:**
- Model: `htdemucs` (balanced)
- Device: `auto` (detects best)
- Output format: `wav` (lossless)
- Output dir: `{input_dir}/{block_name}_stems`

**Philosophy:** Sensible defaults, power when needed.

## Error Handling Flow

### 1. Demucs Not Installed

**Detection:** `shutil.which("demucs")` returns None

**Error Message:**
```
Demucs is not installed or not in PATH.
Install with: pip install demucs
Then ensure 'demucs' command is available in your PATH.
```

**User Action:** Install Demucs, verify PATH

### 2. SSL Certificate Error (macOS)

**Detection:** "SSL" or "certificate" in subprocess output

**Error Message:**
```
SSL certificate verification failed. Demucs needs to download models.
To fix:
1. macOS: Run '/Applications/Python 3.*/Install Certificates.command'
2. Or set SSL_CERT_FILE environment variable to your certificate bundle
3. Or manually download models to ~/.cache/torch/hub/checkpoints/
Original error: [error details]
```

**User Action:** Install certificates (one-time setup)

### 3. Invalid Model Name

**Detection:** Model validation in BlockService

**Error Message:**
```
Invalid Demucs model: 'htdemucs_xyz'

Available Demucs Models:
  htdemucs
    Hybrid Transformer Demucs (fast, good quality)
    Quality: Good | Speed: Fast | Stems: 4
  ...
```

**User Action:** Choose valid model from list

### 4. Input File Not Found

**Detection:** Path validation before subprocess

**Error Message:**
```
Audio file not found: /path/to/file.wav
```

**User Action:** Check file path, ensure LoadAudio processed correctly

### 5. Demucs Subprocess Failure

**Detection:** Non-zero return code from subprocess

**Error Message:**
```
Demucs separation failed: [full subprocess output]
Ensure Demucs is properly installed: pip install demucs
```

**User Action:** Read output, resolve specific issue

### 6. No Stems Produced

**Detection:** Output directory empty after Demucs completes

**Error Message:**
```
Demucs produced no stems in /output/dir
Check that the input file is valid and Demucs completed successfully.
```

**User Action:** Validate input audio file, check Demucs compatibility

## Resource Management

### Memory

**During Execution:**
- Input audio loaded by LoadAudio (already in memory)
- Demucs subprocess has own memory space (isolated)
- Output stems NOT loaded immediately (file_path references)
- Stems loaded on-demand by downstream blocks

**Cleanup:**
- AudioDataItem holds file_path, not audio data
- Temporary files persist until project unload
- Block.cleanup() not needed (no held resources)

### Disk Space

**Per Separation:**
- Input: N MB
- Output (WAV): ~4N MB (4 stems, lossless)
- Output (MP3 320): ~1.5N MB
- Total: 5-6x input size

**Management:**
- User controls output_dir
- Default: Next to input file (user's project space)
- No automatic cleanup (user's output data)

### CPU/GPU

**During Execution:**
- 100% CPU (one core) or 100% GPU utilization
- Duration: Minutes (CPU) to seconds (GPU)
- Blocking operation (sequential execution)

**Impact:**
- No interference with other blocks (sequential)
- Resource released after completion
- No background processing

## Testing Strategy

### Unit Tests

**Processor Validation:**
```python
def test_can_process():
    processor = SeparatorBlockProcessor()
    block = Block(type="Separator", ...)
    assert processor.can_process(block)

def test_cannot_process_other_types():
    processor = SeparatorBlockProcessor()
    block = Block(type="LoadAudio", ...)
    assert not processor.can_process(block)
```

**Device Detection:**
```python
def test_detect_best_device_cuda():
    # Mock torch.cuda.is_available = True
    assert detect_best_device() == "cuda"

def test_detect_best_device_fallback():
    # Mock torch.cuda.is_available = False
    assert detect_best_device() == "cpu"
```

### Integration Tests

**Full Workflow:**
```python
def test_separator_full_workflow(test_audio_file):
    # 1. Create project
    # 2. Add LoadAudio + Separator blocks
    # 3. Connect them
    # 4. Set model
    # 5. Execute
    # 6. Verify stems produced
    # 7. Verify AudioDataItem created
```

**Note:** Requires Demucs installed, may be slow (marked with @pytest.mark.slow)

### Manual Testing

**Checklist:**
- [ ] Different models (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
- [ ] Device modes (auto, cpu, cuda if available)
- [ ] Output formats (wav, mp3)
- [ ] Two-stems mode (vocals, drums, bass, other)
- [ ] Error cases (missing file, invalid model, Demucs not installed)
- [ ] Different audio formats (wav, mp3, flac)

## Code Quality

### Readability
- Clear variable names (model, device, output_dir)
- Extensive comments explaining Demucs specifics
- Logical flow (validate -> build command -> execute -> collect)

### Error Messages
- Specific, actionable guidance
- Common issues anticipated (SSL, PATH)
- Full context provided (file paths, model names)

### Maintainability
- Models defined in DEMUCS_MODELS dict (easy to add more)
- Device detection isolated in function
- Command building is sequential, readable
- No magic numbers (defaults at top)

### Extensibility
- Easy to add new models (update DEMUCS_MODELS)
- Easy to add new options (metadata fields)
- CLI wrapper allows using new Demucs features
- No tight coupling to Demucs internals

## Alignment with Core Values

### "Best Part is No Part"

 **We don't:**
- Implement custom separation algorithms
- Manage model downloads
- Handle device optimization
- Build custom ML infrastructure

 **We do:**
- Thin wrapper around battle-tested tool
- Defer complexity to Demucs
- ~300 lines of clear adapter code

### "Simplicity and Refinement"

 **Simple:**
- CLI subprocess (no complex library integration)
- Defaults work out of box (htdemucs, auto device, wav)
- Clear metadata structure
- Obvious error messages

 **Refined:**
- Auto-detect best device (users don't think about it)
- Streaming output (users see progress)
- Multiple quality/speed options (power users)
- Helpful error guidance (SSL certificates, PATH)

## Future Considerations

### Possible Enhancements (Evaluate Carefully)

**Stem Selection for Downstream Blocks:**
- Allow specifying which stem to use (currently uses first)
- Low complexity, high user value
-  Approve: Simple metadata addition

**Progress Callbacks:**
- Parse Demucs output for progress percentage
- Publish events during execution
- Medium complexity, medium value
- ️ Evaluate: Is it worth parsing complexity?

**Caching/Reuse:**
- Don't re-separate if stems already exist
- Medium complexity, high performance value
- ️ Evaluate: Cache invalidation strategy needed

**Parallel Stem Processing:**
- Allow downstream blocks to process stems in parallel
- High complexity (architectural change)
-  Defer: Requires parallel execution engine

### Features to Avoid

**Embedded Demucs Library:**
-  High coupling
-  Complex integration
-  Maintain CLI wrapper (simpler)

**Custom Model Training:**
-  Massive scope
-  Maintenance nightmare
-  Unlikely to beat Demucs
-  Violates "best part is no part"

**Real-Time Separation:**
-  Demucs not designed for it
-  EchoZero is offline processing tool
-  Would require different architecture

## Summary

The current Demucs integration exemplifies EchoZero's philosophy:

1. **Use the right tool:** Demucs is state-of-the-art, maintained, open source
2. **Keep it simple:** CLI wrapper, not deep integration
3. **Sensible defaults:** Works without configuration
4. **Power when needed:** Advanced options available
5. **Clear errors:** Users know what to do when things fail
6. **Let Demucs be Demucs:** Don't reinvent wheels

This approach makes adding audio separation to workflows trivial while maintaining the flexibility and quality users need.

**Council members: Evaluate proposals against this baseline. Changes should enhance without complicating.**

