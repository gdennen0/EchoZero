# Demucs Module

## Purpose

Provides comprehensive information about Demucs and its integration into EchoZero for audio source separation.

## When to Use

- When evaluating proposals related to audio source separation
- When working with Demucs models or the SeparatorBlock
- When assessing performance or resource requirements for separation
- When making decisions about Demucs integration changes

## Quick Start

1. Read **DEMUCS_OVERVIEW.md** to understand capabilities and models
2. Review **CURRENT_IMPLEMENTATION.md** for architecture details
3. Check **DECISION_CONSIDERATIONS.md** when evaluating proposals
4. Reference **TECHNICAL_DETAILS.md** for system requirements

## Contents

- **DEMUCS_OVERVIEW.md** - What Demucs is, capabilities, available models, quality trade-offs
- **CURRENT_IMPLEMENTATION.md** - How EchoZero integrates Demucs through SeparatorBlock
- **TECHNICAL_DETAILS.md** - Dependencies, hardware requirements, performance, failure modes
- **DECISION_CONSIDERATIONS.md** - Guidance for council members evaluating Demucs proposals
- **OPTIMIZATION_GUIDE.md** - Performance optimizations and best practices
- **README.md** - Additional reference information

## Related Modules

- [`modules/patterns/block_implementation/`](../../patterns/block_implementation/) - When modifying SeparatorBlock implementation
- [`modules/process/council/`](../../process/council/) - For major Demucs-related decisions

## Encyclopedia Links

- [Architecture](../../../docs/ARCHITECTURE.md) - Understanding system architecture
- [Block Implementation](../../patterns/block_implementation/) - Block creation patterns

## Core Values Alignment

This module demonstrates "the best part is no part" through:

**Core Values Context**

When evaluating Demucs-related proposals, remember:

**"The Best Part is No Part"**
- Demucs is a third-party dependency (4.0.0) - 10MB+ dependency justified because audio source separation is a complex domain requiring specialized ML models
- We use the Demucs CLI, not embedding it as a library - simpler integration, clearer separation of concerns
- Model downloads are handled by Demucs itself (~1GB) - we don't manage this complexity

**"Simplicity and Refinement are Key"**
- SeparatorBlock is a thin wrapper around Demucs CLI
- Configuration is minimal but powerful
- Error messages guide users to solutions
- Auto-detection of best device (CPU/GPU)

## Documentation Structure

### [DEMUCS_OVERVIEW.md](./DEMUCS_OVERVIEW.md)
What is Demucs, its capabilities, available models, and quality trade-offs.

**Read this when:**
- Evaluating new model additions
- Understanding what Demucs can/cannot do
- Assessing performance expectations

### [CURRENT_IMPLEMENTATION.md](./CURRENT_IMPLEMENTATION.md)
How EchoZero currently integrates Demucs through the SeparatorBlock.

**Read this when:**
- Reviewing changes to SeparatorBlock
- Understanding current architecture
- Evaluating new separation features

### [TECHNICAL_DETAILS.md](./TECHNICAL_DETAILS.md)
Dependencies, hardware requirements, performance characteristics, failure modes.

**Read this when:**
- Evaluating system-level changes
- Assessing resource requirements
- Planning optimizations

### [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md)
Specific guidance for council members evaluating Demucs-related proposals.

**Read this when:**
- Reviewing any Demucs-related proposal
- Need structured decision framework
- Evaluating alternatives

### [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md)
Performance optimizations, best practices, and user guidance.

**Read this when:**
- Evaluating performance improvement proposals
- Reviewing user-facing documentation
- Assessing new optimization features

## Quick Reference

### What Demucs Does
Separates mixed audio into individual stems (vocals, drums, bass, other instruments) using deep learning.

### Current Integration Status
- **Version:** demucs 4.0.0
- **Integration:** CLI subprocess wrapper in SeparatorBlockProcessor
- **Models Available:** 5 models (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
- **Status:** Fully functional, production-ready

### Key Characteristics
- **External Dependency:** Users must install via pip
- **Model Downloads:** Automatic on first use (~1GB)
- **Processing:** Offline, batch-based (not real-time)
- **Hardware:** CPU or CUDA GPU (MPS not supported due to FFT limitations)
- **Performance:** 30-50x faster on GPU, significant RAM required

### Common Decision Points

**Should we add a new Demucs model?**
-  If model is officially supported by Demucs 4.0.0
-  If model serves different use case (speed vs quality trade-off)
-  If model is experimental/unstable
-  If model duplicates existing capability

**Should we expose more Demucs options?**
-  If option provides clear user value (e.g., two-stems for speed)
-  If option is stable in Demucs API
-  If option is advanced/obscure (cognitive load)
-  If option risks breaking user workflows

**Should we embed Demucs as a library?**
-  Current CLI approach is simpler
-  Would increase coupling and complexity
-  Demucs CLI handles model downloads, updates, device detection
-  Only if CLI approach becomes unmaintainable (unlikely)

**Should we implement custom separation algorithms?**
-  Complex ML domain, high maintenance burden
-  Demucs is state-of-the-art, actively maintained by Meta AI
-  Only if Demucs fundamentally cannot solve user problem

## Integration Philosophy

EchoZero's Demucs integration exemplifies our core values:

1. **Defer to Experts:** Use Demucs for what it's good at (separation), don't reinvent
2. **Simple Wrapper:** Thin adapter layer, not deep integration
3. **User Control:** Expose key options (model, device, output format) but keep defaults simple
4. **Clear Errors:** When Demucs fails, explain why and how to fix
5. **No Magic:** Subprocess call with visible output, user knows what's happening

## For Council Members

When evaluating proposals related to Demucs:

**Architect:** 
- Focus on separation of concerns (CLI vs library)
- Assess coupling between EchoZero and Demucs
- Evaluate abstraction levels

**Systems:**
- Consider resource usage (RAM, CPU, disk)
- Evaluate failure modes and recovery
- Assess dependency risks (Demucs updates, model availability)

**UX:**
- Evaluate user-facing options complexity
- Assess error message clarity
- Consider workflow impact

**Pragmatic:**
- Consider testing complexity (requires Demucs installed)
- Evaluate maintenance burden
- Assess implementation scope

## Common Proposals & Quick Guidance

### "Add GUI for stem visualization"
- Architect:  Separate concern, doesn't affect core
- Systems: ️ Consider memory for loading/visualizing multiple stems
- UX:  High value if done simply
- Pragmatic: ️ Scope carefully, ensure cleanup of loaded audio

### "Cache/reuse separation results"
- Architect:  Good separation of concerns
- Systems:  Reduces redundant processing
- UX:  Faster workflows
- Pragmatic: ️ Consider cache invalidation, disk usage

### "Add more model configuration options"
- Architect:  If done through metadata (current approach)
- Systems:  If Demucs supports it stably
- UX: ️ Each option increases cognitive load
- Pragmatic: ️ More options = more testing, documentation

### "Switch to different separation library"
- Architect: ️ Major change, needs strong justification
- Systems: ️ New dependency risks, stability unknown
- UX: ️ Breaking change for users
- Pragmatic:  High cost, needs compelling benefit

- **Defer to Experts:** Use Demucs CLI instead of embedding, don't reinvent separation
- **Simple Wrapper:** Thin adapter layer, not deep integration
- **No Magic:** Subprocess calls with visible output
- **Justified Dependency:** 10MB+ dependency is acceptable for complex ML domain

The Demucs integration exemplifies simplicity: a thin wrapper around proven technology, avoiding the complexity of custom ML implementations.


