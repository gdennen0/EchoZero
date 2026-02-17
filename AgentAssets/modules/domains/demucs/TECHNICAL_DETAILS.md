# Demucs Technical Details

## Dependencies

### Direct Dependency

**Package:** `demucs==4.0.0`

**Location:** `requirements.txt` (line 29)

**Size:** ~10 MB (package), ~1 GB (models, downloaded on first use)

**License:** MIT (permissive, commercial-friendly)

**Maintenance:** Active (Meta AI, Facebook Research)

### Transitive Dependencies (via Demucs)

Demucs itself depends on:
- **torch** (already required by EchoZero: 2.2.2)
- **torchaudio** (already required: 2.2.2)
- **julius** (audio DSP, small)
- **lameenc** (MP3 encoding, optional)
- **diffq** (quantization support)
- **openunmix** (fallback models)

**Impact:** Minimal - most dependencies already present for other EchoZero features.

### System Dependencies

**Required:**
- Python 3.8+ (EchoZero requirement)
- FFmpeg (audio format support, usually pre-installed)

**Optional:**
- CUDA drivers (for NVIDIA GPU acceleration)
- cuDNN (CUDA Deep Neural Network library)

**Installation:**
```bash
# Basic (CPU)
pip install demucs==4.0.0

# With GPU support (separate step)
# Install CUDA toolkit from NVIDIA
# PyTorch should detect automatically
```

## Hardware Requirements

### Minimum Specs

**CPU-Only:**
- Processor: Modern x86_64 (Intel Core i5 or AMD Ryzen 5 equivalent)
- RAM: 4 GB available (8 GB system recommended)
- Disk: 2 GB free (models + temporary files)
- OS: macOS 10.13+, Linux, Windows 10+

**With GPU:**
- GPU: NVIDIA with 6 GB VRAM (GTX 1060 minimum)
- CUDA: 11.0+ compatible
- RAM: 4 GB available (GPU handles main processing)
- Disk: 2 GB free

### Recommended Specs

**For Good Experience:**
- CPU: Modern 8-core (Intel i7/i9, AMD Ryzen 7/9, Apple M1/M2)
- RAM: 16 GB system (allows multiple operations)
- GPU: NVIDIA RTX 2060 or better (8 GB VRAM)
- Disk: 10 GB free (multiple projects, cached models)

**For Professional Use:**
- CPU: High-end 16+ cores (Ryzen 9, Threadripper, Intel i9)
- RAM: 32 GB+ system
- GPU: NVIDIA RTX 3080+ (10-24 GB VRAM)
- Disk: SSD with 50 GB+ free

### Apple Silicon (M1/M2/M3) Considerations

**Current Status:**
-  Works on CPU mode
-  MPS (Metal Performance Shaders) not supported
-  PyTorch MPS lacks FFT operations (required by Demucs)

**Performance:**
- M1/M2 CPU mode: Comparable to Intel i7 (good, not great)
- M1 Pro/Max/Ultra: Better CPU performance (8-10 cores)
- M3: Similar to M2

**Future:**
- PyTorch may add MPS FFT support (track: pytorch/pytorch#77764)
- If added, would enable ~10x speedup on Apple Silicon
- EchoZero would automatically benefit (auto-detection)

**Recommendation:**
- M1/M2 users: Use CPU mode (automatic), works well
- For heavy Demucs use: Consider cloud GPU or Linux workstation

## Performance Characteristics

### Processing Time

**Variables:**
- Audio length (linear scaling)
- Model choice (htdemucs vs htdemucs_6s)
- Hardware (CPU vs GPU, cores, VRAM)
- Output format (WAV vs MP3)
- Stem mode (4-stem vs 2-stem)

**Benchmark: 4-minute song, htdemucs, 4-stem, WAV output**

| Hardware | Time | Notes |
|----------|------|-------|
| Intel i5 (4-core, 2.5 GHz) | ~15 min | Baseline CPU |
| Intel i7 (8-core, 3.0 GHz) | ~8 min | Better CPU |
| AMD Ryzen 9 (16-core) | ~5 min | High-end CPU |
| Apple M1 (8-core) | ~10 min | Good ARM performance |
| Apple M2 Pro (10-core) | ~7 min | Better ARM |
| NVIDIA GTX 1060 (6 GB) | ~2 min | Entry GPU |
| NVIDIA RTX 2060 (8 GB) | ~1 min | Mid-range GPU |
| NVIDIA RTX 3080 (10 GB) | ~30 sec | High-end GPU |
| NVIDIA RTX 4090 (24 GB) | ~20 sec | Top-tier GPU |

**Scaling:**
- 8-minute song: 2x time
- 16-minute song: 4x time
- Two-stems mode: 0.5x time (2x faster)
- MP3 output: 0.9x time (10% faster)

### Memory Usage

**RAM (CPU Mode):**
- Base: 2 GB (model loaded)
- Processing: +1-2 GB per minute of audio
- Peak: 4-6 GB for 4-minute song
- Scales linearly with audio length

**VRAM (GPU Mode):**
- Base: 2 GB (model on GPU)
- Processing: +0.5-1 GB per minute of audio
- Peak: 3-4 GB for 4-minute song
- More efficient than CPU mode

**Disk I/O:**
- Read: Input audio file (streaming)
- Write: 4x input size (4 stems) or 2x (2 stems)
- Temporary: None (direct write to output)

**Limitations:**
- Very long audio (30+ minutes) may exceed VRAM on smaller GPUs
- Fallback: CPU mode (slower but works)
- Consider: Split long audio, process in chunks

### Parallelization

**Single File:**
- Cannot parallelize (model processes sequentially)
- GPU utilization: ~95-100%
- CPU utilization: 100% of one core (or GPU submission thread)

**Multiple Files:**
- Can process in parallel (separate subprocess per file)
- Limited by: RAM, VRAM, disk I/O
- Recommended: 1 GPU process at a time (VRAM constraint)
- CPU: Can run 2-4 parallel (RAM permitting)

**EchoZero Context:**
- Sequential execution (one block at a time)
- Future: Parallel execution for independent branches
- Current: Not a bottleneck (offline processing)

## Failure Modes

### 1. Demucs Not Installed

**Symptom:** `FileNotFoundError` or "demucs command not found"

**Cause:** Demucs not in PATH or not installed

**Prevention:**
- Installation check during app startup (future enhancement)
- Clear error message with install instructions (current)

**Recovery:**
```bash
pip install demucs==4.0.0
# Verify: demucs --help
```

**Impact:** Blocks execution, user must install

### 2. Model Download Failure

**Symptom:** SSL certificate errors, network timeouts

**Cause:** 
- First-time model download fails
- Network issues
- SSL certificate problems (common on macOS)

**Prevention:**
- Pre-download models (optional setup step)
- Use certifi for SSL bundle (current implementation)

**Recovery:**
```bash
# macOS SSL fix
/Applications/Python\ 3.*/Install\ Certificates.command

# Manual model download
demucs --help  # Triggers download
```

**Impact:** First use only, models cached afterward

### 3. Out of Memory

**Symptom:** Process killed, "CUDA out of memory" error

**Cause:**
- Audio too long for available RAM/VRAM
- Other processes consuming memory

**Prevention:**
- Document memory requirements
- Suggest splitting long audio

**Recovery:**
- Use CPU mode (more RAM available)
- Close other applications
- Split audio into shorter segments
- Use two-stems mode (lower memory)

**Impact:** User must adjust settings or hardware

### 4. Unsupported Audio Format

**Symptom:** Demucs error about file format

**Cause:** Rare or proprietary audio format

**Prevention:**
- LoadAudio validates common formats
- Demucs supports most formats (via FFmpeg)

**Recovery:**
- Convert audio to WAV/MP3 first
- Use FFmpeg externally

**Impact:** Rare, user must pre-process

### 5. Corrupted Audio File

**Symptom:** Demucs fails with decode error

**Cause:** Damaged or incomplete audio file

**Prevention:**
- LoadAudio could validate audio integrity (future)

**Recovery:**
- Repair audio file externally
- Try different audio file

**Impact:** User must fix source audio

### 6. Disk Space Exhausted

**Symptom:** Write error during stem output

**Cause:** Insufficient disk space for output (4x input size)

**Prevention:**
- Check available disk space before processing (future)
- Document space requirements

**Recovery:**
- Free disk space
- Change output_dir to larger volume
- Use MP3 output (smaller)

**Impact:** User must manage disk space

### 7. Process Interrupted

**Symptom:** Partial output, missing stems

**Cause:** User interrupts (Ctrl+C), system crash, timeout

**Prevention:**
- Graceful interrupt handling (future)
- Clear progress indication

**Recovery:**
- Re-run separation (Demucs doesn't resume)
- Check output directory for partial results

**Impact:** Must start over

## Stability Considerations

### Demucs Stability

**Maturity:** High (v4.0.0, years of development)

**Update Frequency:** 
- Major versions: ~1 year
- Patches: As needed
- Risk: Low (stable API)

**Breaking Changes:**
- Rare (CLI interface stable)
- Model names consistent
- Migration path clear

**Recommendation:** Pin version (demucs==4.0.0) until testing new version.

### PyTorch Dependency

**Version:** 2.2.2 (pinned in requirements.txt)

**Compatibility:**
- Demucs 4.0.0 compatible with PyTorch 2.0+
- EchoZero pins 2.2.2 for stability
- Tested combination

**Risk:**
- PyTorch major updates may break compatibility
- Test before upgrading PyTorch

### Model Availability

**Source:** Demucs downloads from official cache

**Cache Location:** `~/.cache/torch/hub/checkpoints/`

**Risk:** 
- Low (models rarely change)
- Network required for first download
- Models persist after download

**Mitigation:**
- Pre-download in setup instructions
- Offline mode possible (if cached)

## Security Considerations

### Subprocess Execution

**Approach:** `subprocess.Popen` with controlled arguments

**Risk:** Command injection via user input

**Mitigation:**
- User input (model name) validated against whitelist
- File paths sanitized (Path objects)
- No shell=True (direct execution)
- Arguments passed as list, not string

**Verdict:** Low risk (good practices followed)

### Downloaded Models

**Source:** Official Demucs/PyTorch repositories

**Risk:** Malicious model replacement (low)

**Mitigation:**
- HTTPS download (with certificate validation)
- Demucs checksums models (internal)
- Use certifi for trusted CA bundle

**Verdict:** Low risk (trusted source)

### File System Access

**Read:**
- User's input audio files
- Controlled by project configuration

**Write:**
- Output directory (user-specified or default)
- Within user's project space

**Risk:** Minimal (no system file access)

**Mitigation:**
- Path validation (no traversal)
- User controls output location

**Verdict:** Low risk (sandboxed to project)

## Monitoring & Observability

### Current Implementation

**Logging:**
```python
Log.info(f"SeparatorBlockProcessor: Separating {input_file} (model={model}, device={device})")
Log.info(f"SeparatorBlockProcessor: Running command: {' '.join(cmd)}")
Log.info(f"SeparatorBlockProcessor: Created stem '{stem_name}' at {stem_path}")
```

**Output Streaming:**
- Real-time subprocess output to console
- User sees Demucs progress
- Errors visible immediately

**Events:**
- ExecutionEngine publishes BlockExecuted event
- Duration tracked
- Success/failure status

### Improvement Opportunities

**Progress Tracking:**
- Parse Demucs output for percentage
- Publish ExecutionProgress events
- Enable progress bars in future GUI

**Resource Metrics:**
- Track memory usage during processing
- Measure disk I/O
- GPU utilization monitoring

**Error Analysis:**
- Categorize failure types
- Track failure rates
- Common issues dashboard

**Performance Profiling:**
- Time per model
- Compare CPU vs GPU times
- Identify bottlenecks

## Optimization Opportunities

### Current Optimizations

 **Auto Device Detection:** Use GPU if available (30-50x speedup)
 **Two-Stems Mode:** 2x faster when one stem needed
 **MP3 Output:** Smaller files, faster writes
 **Streaming Output:** No buffering delay, immediate feedback

### Future Optimizations (Evaluate)

**1. Parallel File Processing**
- Process multiple files simultaneously
- Complexity: Medium (queue management)
- Value: High (batch workflows)
- Risk: VRAM/RAM constraints
- **Verdict:** ️ Worth exploring for batch mode

**2. Result Caching**
- Skip re-separation if stems exist
- Complexity: Medium (cache invalidation)
- Value: High (iterative workflows)
- Risk: Stale results
- **Verdict:** ️ Useful, needs careful design

**3. Adaptive Quality**
- Auto-select model based on hardware
- Complexity: Low (device-based logic)
- Value: Medium (better defaults)
- Risk: Unexpected behavior
- **Verdict:** ️ Consider for future

**4. Chunked Processing**
- Split long audio, process chunks, merge
- Complexity: High (overlap handling, merging)
- Value: Medium (enable GPU for very long audio)
- Risk: Audio artifacts at boundaries
- **Verdict:**  Defer (complex, edge case)

### Anti-Optimizations (Avoid)

 **Custom CUDA Kernels:** Maintain simplicity, trust Demucs
 **Model Quantization (in EchoZero):** Demucs provides mdx_extra_q
 **Custom Model Training:** Out of scope, massive effort
 **Real-Time Streaming:** Not Demucs' design, not EchoZero's use case

## Scalability

### Current Limits

**Single Separation:**
- Audio length: Effectively unlimited (CPU mode)
- GPU mode: ~20-30 minutes (VRAM dependent)
- No hard limit (linear resource scaling)

**Concurrent Separations:**
- Sequential (EchoZero execution model)
- Future: Parallel for independent blocks

**Project Size:**
- No specific limit
- Disk space main constraint

### Bottlenecks

**1. Processing Time (CPU mode)**
- Minutes per song
- Mitigated: GPU acceleration (30-50x)

**2. VRAM (GPU mode)**
- ~4 GB for 4-minute song
- Mitigated: Fall back to CPU for long audio

**3. Disk I/O**
- 4x input size writes
- Mitigated: MP3 output option

**4. Sequential Execution**
- One block at a time
- Mitigated: Future parallel execution

### Horizontal Scaling (Future)

**Distributed Processing:**
- Process files on separate machines
- Low complexity (subprocess isolation)
- High value (professional batch workflows)
- **Status:** Not current need, possible future

**Cloud Integration:**
- Offload to cloud GPUs (AWS, GCP, Azure)
- Medium complexity (API integration)
- High value (access to better hardware)
- **Status:** Consider for future

## Compatibility

### Operating Systems

| OS | Support | Notes |
|----|---------|-------|
| macOS 10.13+ |  Full | CPU mode (MPS pending PyTorch) |
| macOS 11+ (M1/M2) |  Full | CPU mode, good performance |
| Linux (Ubuntu 20.04+) |  Full | Best for CUDA GPU |
| Windows 10/11 |  Full | CUDA supported |
| Other Unix | ️ Likely | Not officially tested |

### Python Versions

| Version | Support | Notes |
|---------|---------|-------|
| Python 3.8 |  Tested | Minimum EchoZero requirement |
| Python 3.9 |  Tested | Recommended |
| Python 3.10 |  Tested | Recommended |
| Python 3.11 |  Works | Less tested |
| Python 3.12+ | ️ Unknown | May work, not guaranteed |

### Audio Formats

**Input (via FFmpeg):**
- WAV 
- MP3 
- FLAC 
- OGG 
- M4A 
- AAC 
- WMA ️ (Windows only)

**Output:**
- WAV (24-bit, 44.1 kHz) 
- MP3 (128/192/320 kbps) 

## Dependencies Risk Assessment

### Demucs (Direct)

**Risk Level:** Low

**Factors:**
-  Stable (v4.0.0, mature)
-  Maintained (Meta AI, active)
-  Open source (MIT license)
-  Large community
-  No controversial dependencies

**Mitigation:** Pin version, test before upgrading

### PyTorch (Transitive)

**Risk Level:** Low-Medium

**Factors:**
-  Industry standard (widely used)
-  Well maintained (Meta AI)
- ️ Large dependency (~1 GB)
- ️ Version compatibility critical

**Mitigation:** Pin version (2.2.2), already required by EchoZero

### FFmpeg (System)

**Risk Level:** Low

**Factors:**
-  Ubiquitous (pre-installed on many systems)
-  Stable (decades old)
- ️ System dependency (not Python package)

**Mitigation:** Installation instructions, check during setup

## For Council Members

### Systems Lens: Key Considerations

**When evaluating Demucs-related proposals:**

1. **Resource Impact:**
   - Will this increase memory usage?
   - Will this increase processing time?
   - Will this increase disk usage?

2. **Stability:**
   - Does this introduce new failure modes?
   - How will errors be handled?
   - What's the recovery path?

3. **Dependencies:**
   - New dependencies needed?
   - Version compatibility checked?
   - Security implications?

4. **Scalability:**
   - How does this scale with audio length?
   - How does this scale with concurrent use?
   - Bottlenecks introduced?

5. **Monitoring:**
   - Can we observe what's happening?
   - Can we debug failures?
   - Are there metrics to track?

### Red Flags

-  "Let's embed Demucs as library" (increases coupling)
-  "Let's implement custom models" (massive complexity)
-  "Let's add real-time separation" (not Demucs' strength)
-  "Let's manage model downloads ourselves" (Demucs does this)

### Green Flags

-  "Let's add progress callbacks" (observability)
-  "Let's cache results" (performance, if done right)
-  "Let's validate disk space first" (failure prevention)
-  "Let's support MPS when PyTorch adds FFT" (free speedup)

## Summary

Demucs integration is technically sound:
- **Stable dependency** (pinned version, mature)
- **Good performance** (GPU acceleration, optimizations)
- **Clear failure modes** (documented, recoverable)
- **Low coupling** (subprocess wrapper)
- **Secure** (validated inputs, trusted sources)
- **Observable** (logging, streaming output)

Council members can confidently evaluate proposals knowing the technical foundation is solid. Focus on whether proposals add value without unnecessary complexity.


