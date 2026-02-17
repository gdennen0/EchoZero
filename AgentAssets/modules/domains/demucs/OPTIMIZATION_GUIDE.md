# Demucs Optimization Guide

## Purpose

This guide documents performance optimizations for Demucs separation in EchoZero, best practices for users, and considerations for future optimization proposals.

## Current Optimization Status

### Implemented Optimizations 

1. **Auto Device Detection**
   - Automatically uses CUDA GPU if available
   - 30-50x speedup over CPU
   - No user configuration needed

2. **Two-Stems Mode**
   - 2x faster when isolating single stem
   - `set_metadata two_stems vocals` enables
   - Outputs target stem + "no_target" mix

3. **MP3 Output Format**
   - Faster writes than WAV
   - ~95% smaller files
   - `set_metadata output_format mp3` enables

4. **MP3 Bitrate Control**
   - Trade quality for size/speed
   - Options: 128, 192, 320 kbps
   - `set_metadata mp3_bitrate 192` enables

5. **Streaming Output**
   - Real-time progress visible
   - No buffering delay
   - User sees Demucs working

6. **Model Selection**
   - mdx_extra_q: Fastest (quantized)
   - htdemucs: Balanced (default)
   - htdemucs_ft: Highest quality

### Not Yet Implemented ⏳

1. **Result Caching**
   - Skip re-separation if stems exist
   - Status: Proposed, not implemented
   - Value: High for iterative workflows

2. **Progress Callbacks**
   - Parse Demucs output for percentage
   - Status: Possible, not implemented
   - Value: Medium (nice for GUI)

3. **Parallel File Processing**
   - Process multiple files simultaneously
   - Status: Deferred (awaits parallel execution engine)
   - Value: High for batch workflows

4. **MPS Support (Apple Silicon)**
   - Use Metal GPU acceleration
   - Status: Blocked by PyTorch (FFT ops unsupported)
   - Value: High (10x speedup on M1/M2)

## Performance Matrix

### Speed vs Quality Trade-offs

| Configuration | Relative Speed | Quality | Use Case |
|--------------|----------------|---------|----------|
| mdx_extra_q + two_stems + MP3 192 | 4x | Good | Fast batch |
| mdx_extra_q + WAV | 2x | Very Good | Balanced fast |
| htdemucs + two_stems | 2x | Good | Fast single stem |
| htdemucs + WAV | 1x | Good | Default |
| htdemucs_ft + WAV | 0.7x | Best | Professional |
| htdemucs_6s + WAV | 0.5x | Best | Detailed 6-stem |

**Baseline:** htdemucs + 4-stem + WAV output on CPU

### Hardware Impact

**CPU Performance (4-minute song, htdemucs):**

| CPU | Cores | Time | Cost |
|-----|-------|------|------|
| Intel i5 (Budget) | 4 | ~15 min | Low |
| Intel i7 (Mid) | 8 | ~8 min | Medium |
| AMD Ryzen 9 (High) | 16 | ~5 min | High |
| Apple M1 (ARM) | 8 | ~10 min | Medium |
| Apple M2 Pro | 10 | ~7 min | High |

**GPU Performance (4-minute song, htdemucs):**

| GPU | VRAM | Time | Cost | Speedup |
|-----|------|------|------|---------|
| GTX 1060 | 6 GB | ~2 min | Low | 7.5x |
| RTX 2060 | 8 GB | ~1 min | Medium | 15x |
| RTX 3080 | 10 GB | ~30 sec | High | 30x |
| RTX 4090 | 24 GB | ~20 sec | Very High | 45x |

**Recommendation:** GPU is single best upgrade for Demucs performance.

## User Best Practices

### For Fast Workflows

**Goal:** Process quickly, quality acceptable

**Configuration:**
```
# Use fast model
sep set_model mdx_extra_q

# Two-stems if only one stem needed
sep set_metadata two_stems vocals

# MP3 output
sep set_metadata output_format mp3
sep set_metadata mp3_bitrate 192
```

**Expected:** 4x faster than default, very good quality

### For Quality Workflows

**Goal:** Maximum quality, time not critical

**Configuration:**
```
# Use best model
sep set_model htdemucs_ft

# WAV output (default)
# All 4 stems (default)
```

**Expected:** ~30% slower than default, best quality

### For Detailed Separation

**Goal:** Individual instrument isolation (6 stems)

**Configuration:**
```
# Use 6-stem model
sep set_model htdemucs_6s

# WAV output for quality
```

**Expected:** 2x slower than default, guitar/piano separated

### For Batch Processing

**Goal:** Process many files efficiently

**Configuration:**
```
# Use fast model
sep set_model mdx_extra_q

# MP3 output to save disk space
sep set_metadata output_format mp3
sep set_metadata mp3_bitrate 192

# Two-stems if applicable
sep set_metadata two_stems vocals
```

**Workflow:**
```bash
# Create template project with optimizations
# Duplicate for each file
# Change input path
# Execute
```

**Future:** Parallel processing will further speed this up.

### For Resource-Constrained Systems

**Goal:** Work within hardware limits

**CPU-Limited (4 GB RAM, old CPU):**
```
# Use fastest model
sep set_model mdx_extra_q

# Two-stems reduces memory
sep set_metadata two_stems vocals

# MP3 output
sep set_metadata output_format mp3
```

**Disk-Limited (SSD with little space):**
```
# MP3 output (much smaller)
sep set_metadata output_format mp3
sep set_metadata mp3_bitrate 192

# Two-stems (half the output files)
sep set_metadata two_stems vocals
```

**VRAM-Limited (GPU with 6 GB, long audio):**
```
# For long audio (20+ minutes), force CPU
sep set_metadata device cpu

# Or split audio into shorter segments
```

## Optimization Scenarios

### Scenario 1: Vocal Extraction for Note Analysis

**Workflow:**
```
LoadAudio -> Separator -> NoteExtractor -> PlotEvents
```

**Optimization:**
```
# Only need vocals
sep set_metadata two_stems vocals

# MP3 sufficient for note extraction
sep set_metadata output_format mp3
sep set_metadata mp3_bitrate 320

# Fast model (note extraction tolerates slight quality loss)
sep set_model mdx_extra_q
```

**Result:** 4x faster, 95% smaller files, negligible quality impact on note extraction

### Scenario 2: Stem Remixing

**Workflow:**
```
LoadAudio -> Separator -> [Manual stem mixing] -> ExportAudio
```

**Optimization:**
```
# All 4 stems needed
# WAV for lossless quality (will be re-mixed)

# Balance speed and quality
sep set_model htdemucs

# Or if time not critical
sep set_model htdemucs_ft
```

**Result:** Best quality for creative work

### Scenario 3: Drum Pattern Analysis

**Workflow:**
```
LoadAudio -> Separator -> DrumClassifier -> PlotEvents
```

**Optimization:**
```
# Only need drums
sep set_metadata two_stems drums

# Fast model
sep set_model mdx_extra_q

# WAV for classification accuracy
```

**Result:** 2x faster, maintains quality for classification

### Scenario 4: Full Separation Archive

**Workflow:**
```
LoadAudio -> Separator -> [Store all stems]
```

**Optimization:**
```
# All stems, best quality
sep set_model htdemucs_ft

# WAV for archival quality

# But use GPU if available (auto-detected)
```

**Result:** Highest quality, fastest on GPU

## Advanced Optimization Techniques

### 1. Custom Output Directory

**Benefit:** Control where stems are saved (e.g., fast SSD)

**Configuration:**
```
sep set_metadata output_dir /path/to/fast/ssd/stems
```

**Use Case:** System disk slow, have faster storage

### 2. Explicit Device Selection

**Benefit:** Override auto-detection (testing, debugging)

**Configuration:**
```
# Force CPU (testing)
sep set_metadata device cpu

# Force CUDA (if auto-detect fails)
sep set_metadata device cuda
```

**Use Case:** Rare, mostly for debugging

### 3. Two-Stems Variations

**Benefit:** Isolate different stems

**Configuration:**
```
# Vocals
sep set_metadata two_stems vocals

# Drums
sep set_metadata two_stems drums

# Bass
sep set_metadata two_stems bass

# Other (all non-vocals/drums/bass)
sep set_metadata two_stems other
```

**Use Case:** When only one instrument needed

## Benchmarking Methodology

### For Council Members Evaluating Performance Claims

**Baseline Configuration:**
- Audio: 4-minute, 44.1 kHz, stereo WAV
- Model: htdemucs (default)
- Mode: 4-stem (default)
- Output: WAV (default)
- Device: Specify (CPU model or GPU model)

**Measurement:**
- Time: Start of execution to completion (wall clock)
- Memory: Peak RAM/VRAM usage
- Disk: Output size
- Quality: Subjective (A/B test) or SDR (objective)

**Valid Comparisons:**
- Change one variable at a time
- Use same hardware
- Use same audio file
- Average 3+ runs (first run includes model download)

**Red Flags:**
- Unrealistic benchmarks (very short audio)
- Cherry-picked results (best case only)
- Unknown hardware (can't reproduce)
- No baseline comparison

## Future Optimization Proposals: Evaluation Guide

### Proposal: Result Caching

**Description:** Don't re-separate if stems already exist

**Analysis:**

**Architect:**
-  Good separation of concerns (cache layer)
- ️ Cache invalidation strategy needed
-  Fits between ExecutionEngine and Processor

**Systems:**
-  Huge performance win (skip expensive processing)
- ️ Disk space (cached stems)
- ️ Cache invalidation rules (input change, model change)

**UX:**
-  Transparent (just faster)
- ️ Could be confusing (why is it fast now?)
- ️ Need "force refresh" option

**Pragmatic:**
- ️ Medium complexity (hash-based caching)
- ️ Edge cases (file modified, model changed)
-  High value for iterative work

**Recommendation:**  APPROVE WITH CONDITIONS
- Implement hash-based cache key (input file hash + model + settings)
- Store cache metadata (when created, settings used)
- Add `--force-refresh` flag to bypass cache
- Document caching behavior

**Implementation Outline:**
```python
def process(block, inputs, metadata):
    cache_key = hash(input_file, model, settings)
    
    if cache_exists(cache_key) and not force_refresh:
        return load_cached_stems(cache_key)
    
    # Run Demucs as usual
    result = run_demucs(...)
    
    cache_result(cache_key, result)
    return result
```

### Proposal: Progress Callbacks

**Description:** Parse Demucs output for progress percentage, publish events

**Analysis:**

**Architect:**
-  Fits event system (ExecutionProgress events)
-  No architecture changes
- ️ Parsing logic couples to Demucs output format

**Systems:**
-  Improved observability
- ️ Fragile (Demucs output format changes)
-  Graceful degradation (if parsing fails, no harm)

**UX:**
-  High value (users see progress)
-  Especially valuable in future GUI
-  Reduces "is it frozen?" confusion

**Pragmatic:**
- ️ Medium complexity (regex parsing)
- ️ Brittle (output format dependency)
-  Can be optional (fallback to no progress)

**Recommendation:**  APPROVE WITH CONDITIONS
- Parse Demucs output (regex or split)
- Publish ExecutionProgress events (already exists)
- Gracefully handle parsing failures (don't break execution)
- Test with multiple Demucs versions (ensure compatibility)

**Implementation Outline:**
```python
# In subprocess output streaming
for line in process.stdout:
    # Try to extract progress
    match = re.search(r'(\d+)%', line)
    if match:
        progress = int(match.group(1))
        event_bus.publish(ExecutionProgress(block_id, progress))
    
    # Always log line (progress parsing is bonus)
    print(line)
```

### Proposal: Parallel File Processing

**Description:** Process multiple audio files simultaneously

**Analysis:**

**Architect:**
- ️ Affects ExecutionEngine (currently sequential)
- ️ Need parallel execution design first
-  Separator itself is stateless (good for parallel)

**Systems:**
-  Better hardware utilization
-  Resource contention (RAM, VRAM, disk)
- ️ Need resource limits (max parallel)
- ️ Need queue management

**UX:**
-  Faster batch processing
- ️ Complex error handling (partial failures)
- ️ Progress display more complex

**Pragmatic:**
-  High complexity (requires parallel execution engine)
- ️ Defer until parallel execution designed
-  High value when implemented

**Recommendation:** ️ DEFER
- Depends on broader parallel execution design
- Not specific to Demucs
- Should be handled at ExecutionEngine level
- Revisit when parallel execution is designed

### Proposal: Custom CUDA Kernels

**Description:** Write custom GPU code for separation

**Analysis:**

**Architect:**
-  Violates "defer to experts"
-  Deep coupling to ML internals
-  Replaces Demucs (not enhances)

**Systems:**
-  Massive maintenance burden
-  CUDA version compatibility
-  GPU-specific code (limits portability)
-  Unlikely to beat Demucs (years of research)

**UX:**
- ️ Only better if faster/higher quality
-  Unlikely to improve both
-  Risk of regressions

**Pragmatic:**
-  Massive scope (months of work)
-  Requires ML expertise
-  Ongoing maintenance nightmare
-  High risk of failure

**Recommendation:**  REJECT
- Violates "best part is no part"
- Demucs is state-of-the-art by Meta AI
- We cannot realistically improve on it
- Focus on integration quality, not reimplementation

## Performance Troubleshooting

### Issue: Separation is Very Slow

**Diagnosis:**
1. Check device: Is GPU being used?
   ```
   # Look for log: "Auto-detected device: cuda" or "cpu"
   ```
2. Check hardware: CPU cores, available RAM
3. Check model: Is it the slowest (htdemucs_6s)?

**Solutions:**
- If CPU used but GPU available: Check CUDA installation
- If slow CPU: Use mdx_extra_q model, two-stems mode
- If htdemucs_6s: Switch to htdemucs for speed
- If long audio: Consider splitting

### Issue: Out of Memory

**Diagnosis:**
1. Check audio length (very long?)
2. Check VRAM (GPU) or RAM (CPU)
3. Check concurrent processes

**Solutions:**
- Force CPU mode (more RAM available)
- Use two-stems mode (lower memory)
- Close other applications
- Split audio into shorter segments

### Issue: Quality is Poor

**Diagnosis:**
1. Check input audio quality (low bitrate MP3?)
2. Check model (mdx_extra_q is fastest but lower quality)
3. Check output format (MP3 loses quality)

**Solutions:**
- Use lossless input (WAV, FLAC)
- Use htdemucs_ft model (best quality)
- Use WAV output
- Accept trade-off (speed vs quality)

### Issue: Output Files Too Large

**Diagnosis:**
1. WAV output (lossless, large)
2. All 4 stems (4x input size)

**Solutions:**
- Use MP3 output (320 kbps still very good)
- Use two-stems if only one needed
- Compress after separation (external tool)

## Summary: Optimization Philosophy

**Current Approach:**
- Sensible defaults (htdemucs, auto device, WAV)
- Power options available (models, two-stems, MP3)
- Let Demucs handle complexity
- Focus on configuration, not reimplementation

**Future Direction:**
- Add caching (high value, medium complexity)
- Add progress parsing (medium value, medium complexity)
- Wait for MPS support (free upgrade when PyTorch ready)
- Defer parallel processing (needs broader design)

**What We Don't Do:**
- Custom ML models (violates "best part is no part")
- Low-level GPU optimization (trust Demucs)
- Real-time separation (not Demucs' design)
- Automatic quality/speed selection (user controls trade-off)

**For Council Members:**

Approve optimizations that:
-  Make common cases faster without configuration
-  Provide clear control for trade-offs
-  Maintain simplicity and boundaries
-  Add value > cost

Reject optimizations that:
-  Increase coupling to Demucs internals
-  Reinvent what Demucs already does
-  Add complexity without clear benefit
-  Sacrifice simplicity for marginal gains

**The goal is fast, high-quality separation with simple, intuitive controls. Current implementation achieves this.**


