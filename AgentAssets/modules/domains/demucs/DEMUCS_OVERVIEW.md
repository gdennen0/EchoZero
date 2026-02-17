# Demucs Overview

## What is Demucs?

Demucs is an advanced deep learning model developed by **Meta AI** for music source separation. It enables extraction of individual audio components (vocals, drums, bass, other instruments) from a mixed audio track using state-of-the-art neural networks.

### Key Characteristics

**Technology:**
- Hybrid Transformer architecture (version 4.0+)
- U-Net convolutional structure with bidirectional LSTM
- Operates directly on waveforms (not spectrograms)
- Trained on large-scale music datasets

**Capabilities:**
- 4-stem separation: vocals, drums, bass, other
- 6-stem separation: vocals, drums, bass, other, guitar, piano
- 2-stem separation: isolate one stem, combine rest (2x faster)
- Multiple quality/speed trade-offs

**Performance:**
- State-of-the-art on MusDB18 benchmark (SDR 6.3-6.8)
- Surpasses some oracle methods for specific instruments
- GPU acceleration: 30-50x faster than CPU
- Natural-sounding results (waveform-based processing)

## Available Models

EchoZero supports 5 Demucs models with different trade-offs:

### htdemucs (Default)
```
Quality: Good
Speed: Fast
Stems: 4 (vocals, drums, bass, other)
```

**Best for:**
- General-purpose separation
- Balanced quality/speed
- Most common use case

**Characteristics:**
- Hybrid Transformer architecture
- Fast processing
- Good quality for most music

### htdemucs_ft (Best Quality)
```
Quality: Best
Speed: Slower
Stems: 4 (vocals, drums, bass, other)
```

**Best for:**
- Professional/production work
- Maximum quality needed
- When processing time is not critical

**Characteristics:**
- Fine-tuned on additional data
- Highest quality separation
- Slower processing (worth it for quality)

### htdemucs_6s (6-Stem)
```
Quality: Best
Speed: Slowest
Stems: 6 (vocals, drums, bass, other, guitar, piano)
```

**Best for:**
- Detailed instrument separation
- Music with guitar or piano
- Advanced remixing/analysis

**Characteristics:**
- Separates guitar and piano from "other"
- Best quality, most detailed
- Highest computational cost

### mdx_extra (Alternative)
```
Quality: Very Good
Speed: Medium
Stems: 4 (vocals, drums, bass, other)
```

**Best for:**
- Alternative to htdemucs
- Different separation characteristics
- When htdemucs doesn't work well

**Characteristics:**
- MDX (Music Demixing Challenge) architecture
- Different strengths than htdemucs
- Good alternative approach

### mdx_extra_q (Quantized)
```
Quality: Very Good
Speed: Fast
Stems: 4 (vocals, drums, bass, other)
```

**Best for:**
- Faster processing with good quality
- Resource-constrained systems
- Batch processing

**Characteristics:**
- Quantized version of mdx_extra
- Faster than mdx_extra
- Similar quality with reduced resources

## Quality vs Speed Trade-offs

### For Speed Priority:
```
1. mdx_extra_q      (fastest, very good quality)
2. htdemucs         (fast, good quality)
3. mdx_extra        (medium, very good quality)
```

### For Quality Priority:
```
1. htdemucs_ft      (best quality, slower)
2. htdemucs_6s      (best quality, 6 stems, slowest)
3. mdx_extra        (very good quality, medium)
```

### For Specific Needs:
- **Guitar/piano separation:** htdemucs_6s (only option)
- **Fast batching:** mdx_extra_q + two-stems mode
- **Balanced:** htdemucs (default)

## Performance Characteristics

### Processing Time (Approximate)

For a 4-minute song:

**CPU (Modern Intel/AMD):**
- mdx_extra_q: ~8-10 minutes
- htdemucs: ~10-15 minutes
- htdemucs_ft: ~15-20 minutes
- htdemucs_6s: ~20-30 minutes

**CUDA GPU (NVIDIA RTX 3080):**
- mdx_extra_q: ~20-30 seconds
- htdemucs: ~30-45 seconds
- htdemucs_ft: ~45-60 seconds
- htdemucs_6s: ~60-90 seconds

**Two-Stems Mode:**
- 2x faster than full 4-stem separation
- Only outputs target stem + "no_{stem}"

### Memory Requirements

**RAM:**
- CPU: 4-8 GB minimum, 16 GB recommended
- GPU: 6 GB VRAM minimum, 8 GB+ recommended

**Disk:**
- Model cache: ~1 GB (downloaded once)
- Output files: ~4x input size (4 stems, lossless)
- With MP3 output: ~1.5x input size

**Scalability:**
- Linear with audio length
- Independent per-file (parallelizable)
- GPU memory limits long files (20+ minutes may need CPU)

## Audio Quality Considerations

### Input Format Impact

**Lossless (Recommended):**
- WAV, FLAC, AIFF
- Preserves high-frequency details
- Best separation quality
- Supported: 

**Lossy (Acceptable):**
- MP3 320kbps: Good results
- MP3 192kbps: Acceptable results
- MP3 128kbps or lower: Degraded quality
- Supported: 

**Avoid:**
- Heavily compressed audio
- Low bitrate streaming rips
- Already-separated stems (no improvement)

### Output Format Trade-offs

**WAV (Default):**
- Lossless quality
- Large file size (~40 MB per minute per stem)
- Best for further processing
- Recommended for professional work

**MP3 320kbps:**
- Near-transparent quality
- 90% smaller than WAV
- Good for distribution
- Slight quality loss

**MP3 192kbps:**
- Audible compression
- 95% smaller than WAV
- Acceptable for casual use
- Noticeable quality loss

## Use Cases

### Current EchoZero Use Cases

1. **Vocal Isolation**
   - Extract vocals for pitch analysis
   - Remove vocals for karaoke
   - Vocal effect processing

2. **Drum Analysis**
   - Isolate drums for pattern detection
   - Drum classification workflows
   - Rhythm analysis

3. **Instrument Isolation**
   - Bass extraction
   - Guitar/piano separation (6s model)
   - Instrument-specific processing

4. **Remixing**
   - Separate stems for remixing
   - Volume balancing per instrument
   - Creative stem manipulation

5. **Music Analysis**
   - Note extraction per instrument
   - Onset detection per stem
   - Multi-track visualization

### Outside EchoZero Scope

**Not Suitable For:**
- Real-time separation (offline processing only)
- Low-latency applications (minutes per song)
- Live performance (not designed for it)
- Speech separation (optimized for music)
- Restoration of very poor quality audio

**Better Alternatives For:**
- Speech/voice separation: Spleeter, OpenUnmix
- Real-time: Plugin-based solutions (iZotope, Audionamix)
- Speech enhancement: Specialized voice ML models

## Technical Background

### Architecture Evolution

**Demucs v1 (2019):**
- Pure time-domain separation
- U-Net architecture
- Good quality, slow

**Demucs v2 (2020):**
- Improved architecture
- Better quality
- Still slow

**Demucs v3 (2021):**
- Hybrid approach (time + frequency)
- Significant speed improvement
- Better quality

**Demucs v4 (2023) - Current:**
- Hybrid Transformer models
- State-of-the-art quality
- Multiple model variants
- Best balance of speed/quality

### Why Demucs?

**Advantages:**
1. **State-of-the-art quality:** Best publicly available separation
2. **Active development:** Maintained by Meta AI
3. **Production-ready:** Stable, well-tested
4. **Comprehensive:** Multiple models, trade-offs
5. **Open source:** MIT license, transparent
6. **Community:** Large user base, well-documented

**Limitations:**
1. **Large models:** ~1GB download
2. **Computational:** Requires significant resources
3. **Offline only:** Not real-time capable
4. **Music-focused:** Not optimized for speech
5. **GPU dependency:** CPU is very slow

### Alternatives Considered

**Spleeter (Deezer):**
-  Less maintained (last update 2020)
-  Lower quality than Demucs v4
-  Slightly faster
- **Verdict:** Demucs is better choice

**OpenUnmix:**
-  Lower quality
-  Lighter weight
-  Fewer features
- **Verdict:** Demucs provides better value

**Commercial (iZotope RX, Audionamix):**
-  Similar or better quality
-  Expensive licensing
-  Not open source
-  Not scriptable
- **Verdict:** Not suitable for EchoZero

**Custom Models:**
-  Massive development cost
-  Maintenance burden
-  Unlikely to match Demucs quality
- **Verdict:** Not justified

## Model Selection Guide

### Decision Tree

```
Do you need guitar or piano separated individually?
├─ YES → htdemucs_6s (only option)
└─ NO → Continue

Is processing speed critical?
├─ YES → Do you need high quality?
│   ├─ YES → mdx_extra_q (fast + good quality)
│   └─ NO → htdemucs + two_stems (fastest)
└─ NO → Continue

Is this for professional/production use?
├─ YES → htdemucs_ft (best quality)
└─ NO → htdemucs (balanced default)
```

### Recommendation by User Type

**Casual Users:**
- Default: `htdemucs`
- Fast: `htdemucs` + two_stems mode

**Power Users:**
- Quality: `htdemucs_ft`
- Speed: `mdx_extra_q`

**Professional:**
- Best: `htdemucs_ft` + WAV output
- Detailed: `htdemucs_6s` + WAV output

**Developers/Batch:**
- Balanced: `htdemucs` + MP3 output
- Fast: `mdx_extra_q` + two_stems + MP3

## Benchmarks & Quality Metrics

### MusDB18 Dataset (Standard Benchmark)

**Signal-to-Distortion Ratio (SDR):**
- htdemucs_ft: 6.8 SDR (best public model)
- htdemucs: 6.5 SDR
- htdemucs_6s: 6.7 SDR
- mdx_extra: 6.4 SDR

**Context:**
- Higher SDR = better separation
- 6.8 SDR is state-of-the-art for open models
- Some oracle methods: 6.5-7.0 SDR
- Human perception: Differences noticeable above 6.0 SDR

### Real-World Performance

**Subjective Quality:**
- Vocals: Excellent (cleanest separation)
- Drums: Very Good (occasional bleed in busy mixes)
- Bass: Good (can have mid-range bleed)
- Other: Variable (catch-all for many instruments)

**Genre Performance:**
- Pop/Rock: Excellent
- Electronic/EDM: Very Good
- Classical: Good (complex instrumentation challenges)
- Jazz: Good (many instruments in "other")
- Acoustic: Excellent

## For Council Members

### When Evaluating Model Additions

**Ask:**
- Is this model officially supported by Demucs 4.0.0?
- Does it provide clear user benefit (quality, speed, stems)?
- Is it stable and maintained?
- Does it fit existing model selection pattern?

**Approve if:**
- Official Demucs model
- Clear differentiation from existing models
- Stable/production-ready

**Reject if:**
- Experimental/beta model
- Marginal difference from existing model
- Stability concerns

### When Evaluating Demucs Alternatives

**Ask:**
- What specific problem does this solve that Demucs doesn't?
- What is the quality/maintenance/cost trade-off?
- Does this align with "best part is no part"?

**High bar for approval:**
- Demucs is state-of-the-art and maintained
- Switching has high cost (code changes, testing, user disruption)
- Need compelling evidence of significant benefit

## Resources

**Official Documentation:**
- GitHub: https://github.com/facebookresearch/demucs
- PyPI: https://pypi.org/project/demucs/
- Paper: https://arxiv.org/abs/1911.13254

**EchoZero Implementation:**
- SeparatorBlock: `src/application/blocks/separator_block.py`
- Block Registry: `src/application/block_registry.py` (lines 427-451)
- Tests: Search codebase for "separator" or "demucs"

**User Documentation:**
- Commands: `docs/COMMAND_FEATURES.md`
- Examples: `docs/NOTE_EXTRACTION_GUIDE.md`, `docs/PLOT_EVENTS_GUIDE.md`


