# Demucs Integration Documentation

## Quick Start

This folder contains comprehensive documentation about Demucs and its integration into EchoZero. If you're an AI agent council member evaluating a Demucs-related proposal, start here.

## What is Demucs?

Demucs is a state-of-the-art AI audio source separation system developed by Meta AI. It separates mixed music into individual stems (vocals, drums, bass, other instruments) using deep learning.

**In EchoZero:** Demucs is integrated as a **thin CLI wrapper** in the SeparatorBlock, following our "best part is no part" philosophy - we defer to the experts and maintain simplicity.

## Navigation

### Start Here: [INDEX.md](./INDEX.md)
Complete table of contents with quick reference decision points. Read this first to understand the documentation structure.

### Essential Reading for Council Members

1. **Understanding Demucs**: [DEMUCS_OVERVIEW.md](./DEMUCS_OVERVIEW.md)
   - What Demucs does and how it works
   - Available models and trade-offs
   - Performance characteristics
   - When to use (and not use) Demucs

2. **How We Integrate It**: [CURRENT_IMPLEMENTATION.md](./CURRENT_IMPLEMENTATION.md)
   - Architecture (CLI wrapper approach)
   - Component breakdown
   - Error handling
   - Alignment with core values

3. **Technical Deep Dive**: [TECHNICAL_DETAILS.md](./TECHNICAL_DETAILS.md)
   - Dependencies and requirements
   - Hardware considerations
   - Performance characteristics
   - Failure modes and stability

4. **Decision Making**: [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md)
   - Structured guidance for proposals
   - Decision trees for common scenarios
   - Council member analysis templates
   - Red flags and green flags

5. **Performance**: [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md)
   - Current optimizations
   - Best practices for users
   - Future optimization proposals
   - Benchmarking methodology

## Quick Reference

### Current Status
- **Version:** demucs 4.0.0
- **Integration:** CLI subprocess wrapper (SeparatorBlockProcessor)
- **Models:** 5 available (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
- **Performance:** 30-50x faster on GPU, works on CPU
- **Status:** Production-ready, stable

### Core Values Alignment

 **"Best Part is No Part"**
- Use Demucs CLI (don't reinvent separation)
- ~300 lines of thin wrapper code
- Defer model management to Demucs
- No custom ML infrastructure

 **"Simplicity and Refinement"**
- Works out of box (sensible defaults)
- Clear error messages with guidance
- Power options available when needed
- Explicit subprocess call (no magic)

### Common Decisions

| Proposal | Default Answer | See |
|----------|----------------|-----|
| Add official Demucs model |  Likely Approve | [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md#1-adding-new-demucs-model) |
| Expose new Demucs option | ️ Evaluate Carefully | [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md#2-exposing-new-demucs-options) |
| Switch from CLI to library |  Reject | [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md#3-changing-integration-approach) |
| Custom separation algorithm |  Reject | [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md#4-custom-separation-features) |
| Result caching |  Approve w/ Conditions | [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md#proposal-result-caching) |
| Progress callbacks |  Approve w/ Conditions | [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md#proposal-progress-callbacks) |
| Parallel processing | ️ Defer | [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md#proposal-parallel-file-processing) |

## For Different Council Members

### Architect
**Focus Areas:**
- Separation of concerns (CLI wrapper vs library)
- Coupling between EchoZero and Demucs
- Abstraction levels and boundaries

**Key Files:**
- [CURRENT_IMPLEMENTATION.md](./CURRENT_IMPLEMENTATION.md) - Architecture approach
- [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md) - Analysis templates

### Systems
**Focus Areas:**
- Resource usage (RAM, CPU/GPU, disk)
- Failure modes and stability
- Dependency risks and management

**Key Files:**
- [TECHNICAL_DETAILS.md](./TECHNICAL_DETAILS.md) - Dependencies, performance, failure modes
- [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md) - Performance characteristics

### UX
**Focus Areas:**
- User-facing options and complexity
- Error message clarity
- Workflow impact and usability

**Key Files:**
- [DEMUCS_OVERVIEW.md](./DEMUCS_OVERVIEW.md) - Models, use cases, quality
- [CURRENT_IMPLEMENTATION.md](./CURRENT_IMPLEMENTATION.md) - User workflows

### Pragmatic
**Focus Areas:**
- Implementation complexity
- Testing requirements
- Maintenance burden

**Key Files:**
- [CURRENT_IMPLEMENTATION.md](./CURRENT_IMPLEMENTATION.md) - Code structure, testing
- [DECISION_CONSIDERATIONS.md](./DECISION_CONSIDERATIONS.md) - Pragmatic analysis template

## Evaluation Checklist

When reviewing a Demucs-related proposal, ask:

### Core Values
- [ ] Does this maintain the CLI wrapper approach?
- [ ] Does this defer complexity to Demucs?
- [ ] Does this keep the integration simple?
- [ ] Does this respect user hardware constraints?

### Architecture
- [ ] Does this maintain separation between EchoZero and Demucs?
- [ ] Does this fit the existing BlockProcessor pattern?
- [ ] Does this avoid tight coupling to Demucs internals?

### Systems
- [ ] Does this introduce new failure modes?
- [ ] Does this increase resource usage significantly?
- [ ] Does this add dependencies?
- [ ] Is error handling adequate?

### UX
- [ ] Does this make common tasks easier?
- [ ] Does this increase cognitive load?
- [ ] Are error messages clear and actionable?
- [ ] Does this maintain consistency?

### Pragmatic
- [ ] Is the scope reasonable?
- [ ] Is this testable?
- [ ] What's the maintenance burden?
- [ ] Is this reversible?

## Red Flags

**Immediate Concerns:**
-  "Embed Demucs as library" → Violates simplicity
-  "Custom ML model training" → Massive scope
-  "Real-time separation" → Not Demucs' design
-  "Expose all options" → Cognitive overload
-  "No defaults needed" → Unusable
-  "We might need this someday" → Premature

## Green Flags

**Positive Indicators:**
-  Maintains CLI wrapper approach
-  Improves error handling/messages
-  Adds observability (progress, logging)
-  Makes common cases simpler
-  Provides power when needed
-  Reduces resource usage
-  Fits existing patterns

## Example Scenarios

### Scenario 1: "Add htdemucs_7s model"

**Quick Check:**
1. Is this an official Demucs model? → Check Demucs docs
2. Does it provide clear differentiation? → 7 stems vs 4/6
3. Is it stable? → Check Demucs release status

**If YES to all:**  Likely approve (update DEMUCS_MODELS dict)
**If NO to any:**  Reject or defer

### Scenario 2: "Add progress bar for separation"

**Quick Check:**
1. Maintains CLI wrapper? → Yes (just parse output)
2. Improves UX? → Yes (visibility)
3. Adds complexity? → Medium (parsing logic)
4. Breaks if Demucs changes output? → Maybe (fragile)

**Decision:**  Approve with conditions (graceful degradation)

### Scenario 3: "Switch to Spleeter library"

**Quick Check:**
1. Maintains CLI wrapper? → No (different integration)
2. Is Demucs broken? → No
3. Is Spleeter better? → No (Demucs is state-of-the-art)
4. Breaking change? → Yes (users affected)

**Decision:**  Reject (no compelling benefit, high cost)

## Integration Philosophy

EchoZero's Demucs integration is a success story for our values:

**We Use Demucs For:**
- State-of-the-art separation (we can't do better)
- Model management (downloads, caching)
- Device optimization (CUDA detection)
- Audio format handling (via FFmpeg)

**We Handle:**
- EchoZero workflow integration (blocks, connections)
- User-facing configuration (simple metadata)
- Error messages and guidance (make failures clear)
- Resource awareness (respect user's system)

**We Don't Do:**
- Custom ML models (defer to experts)
- Model training (out of scope)
- Low-level optimization (trust Demucs)
- Real-time separation (not the design)

## Summary

When evaluating Demucs proposals, remember:

1. **Current integration is good** - Most changes should be refinements
2. **Maintain simplicity** - CLI wrapper is correct approach
3. **Respect boundaries** - Don't increase coupling
4. **User value first** - Does this solve real problems?
5. **Evidence required** - Claims need benchmarks/proof

The best improvements make the current approach more robust, observable, and user-friendly without adding unnecessary complexity.

## Updates

- **November 26, 2025:** Initial documentation created
- **Future:** Update when Demucs version changes, new models added, or PyTorch adds MPS FFT support

## Related Documentation

- **Main AgentAssets:** [../../../README.md](../../../README.md)
- **Project Overview:** [../../../docs/README.md](../../../docs/README.md)
- **Core Values:** [../../../CORE_VALUES.md](../../../CORE_VALUES.md)
- **Decision Framework:** [../../process/council/FRAMEWORK.md](../../process/council/FRAMEWORK.md)
- **Technical Architecture:** [../../../docs/architecture/ARCHITECTURE.md](../../../docs/architecture/ARCHITECTURE.md)

## Questions?

If these docs don't answer your question:

1. Check the [INDEX.md](./INDEX.md) for specific topics
2. Search the main EchoZero codebase (`src/application/blocks/separator_block.py`)
3. Review Demucs official documentation (https://github.com/facebookresearch/demucs)
4. Ask for clarification (gaps in documentation should be filled)

---

**Remember: The best part is no part. Keep Demucs integration simple, defer to the experts, and focus on making it excellent at what it does.**


