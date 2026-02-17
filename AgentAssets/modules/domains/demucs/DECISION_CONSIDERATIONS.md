# Decision Considerations for Demucs Proposals

## Purpose

This document provides structured guidance for council members evaluating proposals related to Demucs, audio separation, or the SeparatorBlock. Use this alongside the general [Decision Framework](../../process/council/FRAMEWORK.md).

## Demucs-Specific Context

### What Makes Demucs Special?

1. **External ML System:** Deep learning models, not simple algorithms
2. **Resource Intensive:** Significant CPU/GPU, memory, time
3. **Third-Party Dependency:** We don't control Demucs internals
4. **Complex Domain:** Audio source separation is ML research area
5. **CLI Integration:** Subprocess wrapper, not embedded library

### Core Principles for Demucs Decisions

1. **Defer to Experts:** Demucs is state-of-the-art, don't reinvent
2. **Maintain Separation:** Thin wrapper, clear boundaries
3. **User Control with Defaults:** Power available, simplicity default
4. **Clear Error Guidance:** ML can fail, users need direction
5. **Resource Awareness:** Processing is expensive, respect user's hardware

## Decision Tree for Common Proposals

### 1. Adding New Demucs Model

```
Is the model officially part of Demucs 4.0.0?
├─ NO →  Reject (stability risk, maintenance)
└─ YES → Continue

Does it provide clear differentiation?
├─ NO (similar to existing) →  Reject (unnecessary complexity)
└─ YES → Continue

What trade-off does it offer?
├─ Quality/Speed trade-off →  Consider (user value)
├─ More stems (e.g., 6s vs 4s) →  Consider (new capability)
└─ Marginal improvement → ️ Evaluate carefully

Implementation effort?
├─ Just add to DEMUCS_MODELS dict →  Low cost
├─ Requires code changes → ️ Evaluate complexity
└─ Requires new features →  Reconsider scope
```

**Example: htdemucs_6s**
-  Official Demucs model
-  Clear differentiation (6 stems vs 4)
-  New capability (guitar/piano separation)
-  Simple addition (dict entry)
- **Verdict:** APPROVE

**Example: experimental_model_xyz**
-  Not in official release
-  Stability unknown
-  May break in future
- **Verdict:** REJECT

### 2. Exposing New Demucs Options

```
Is the option supported by Demucs CLI?
├─ NO →  Reject (can't implement)
└─ YES → Continue

Does it solve a real user problem?
├─ NO ("nice to have") →  Reject
└─ YES → Continue

Does it increase cognitive load?
├─ High (obscure, technical) →  Reject
├─ Medium (power user feature) → ️ Evaluate carefully
└─ Low (clear value) →  Consider

Can we provide good defaults?
├─ NO (user must configure) → ️ Reconsider
└─ YES (works without config) →  Good

What's the complexity cost?
├─ Metadata field only →  Low cost
├─ UI changes needed → ️ Medium cost
├─ Architecture changes →  High cost, reconsider
```

**Example: two_stems option**
-  Supported by Demucs (--two-stems)
-  Solves problem (2x faster, isolate one stem)
-  Low cognitive load (clear purpose)
-  Good default (None = full 4-stem)
-  Metadata field only
- **Verdict:** APPROVE (already implemented)

**Example: --jobs parallel option**
-  Supported by Demucs
- ️ Complex (parallel processing)
- ️ Resource management concerns
- ️ May conflict with EchoZero execution
-  High complexity for questionable benefit
- **Verdict:** REJECT (or defer to parallel execution design)

### 3. Changing Integration Approach

```
What's the proposed change?
├─ CLI wrapper → Embedded library
│   ├─ Why? "More control"
│   ├─ Cost: High coupling, complexity
│   └─ Verdict:  REJECT (violates simplicity)
│
├─ CLI wrapper → Different library (Spleeter, etc.)
│   ├─ Why? "Better performance/quality"
│   ├─ Evidence? Benchmark required
│   ├─ Breaking change? YES (user impact)
│   ├─ Maintenance? Different dependency
│   └─ Verdict: ️ High bar (need compelling evidence)
│
└─ Improve CLI wrapper
    ├─ Why? "Better error handling, progress, etc."
    ├─ Cost: Low to medium
    ├─ Value: Incremental improvement
    └─ Verdict:  CONSIDER (aligns with refinement)
```

**Example: "Switch from CLI to library"**
-  Violates "simplicity" (increases coupling)
-  Violates "best part is no part" (more integration code)
-  Questionable benefit (CLI works fine)
-  Higher maintenance (tied to library internals)
- **Verdict:** REJECT

**Example: "Add progress parsing"**
-  Improves UX (visibility)
-  Uses existing CLI output
-  Low coupling (parse output, don't change integration)
- ️ Parsing complexity (Demucs output format changes?)
- **Verdict:** APPROVE WITH CONDITIONS (ensure robust parsing)

### 4. Custom Separation Features

```
What's being proposed?
├─ Custom ML model training
│   └─  REJECT (massive scope, violates "best part is no part")
│
├─ Pre/post-processing in EchoZero
│   ├─ Examples: Normalization, filtering, mixing
│   ├─ Value? Enhances separation quality
│   ├─ Complexity? Additional processing blocks
│   └─ ️ EVALUATE (may be better as separate blocks)
│
└─ Separation parameter tuning
    ├─ Examples: Overlap, segment size
    ├─ Demucs supports? Check CLI options
    ├─ User value? Advanced optimization
    └─ ️ EVALUATE (power user vs cognitive load)
```

**Example: "Train custom model on user's music"**
-  Massive scope (ML pipeline, training infrastructure)
-  Violates core values (complexity explosion)
-  Unlikely to beat Demucs (trained on large datasets)
-  Maintenance nightmare (model versioning, updates)
- **Verdict:** REJECT

**Example: "Add stem mixing block"**
-  Separate concern (not Separator's job)
-  Simple (combine AudioDataItems)
-  Flexible (use with any stems, not just Demucs)
-  Fits block architecture
- **Verdict:** APPROVE AS SEPARATE BLOCK

## Council Member Analysis Templates

### Architect Lens: Demucs Proposal Template

```
PROPOSAL: [Name]

ARCHITECTURAL ANALYSIS:

1. Boundaries and Coupling:
   - Does this maintain separation between EchoZero and Demucs?
   - Does this increase coupling to Demucs internals?
   - Does this respect layer boundaries (application/domain/infrastructure)?

2. Abstraction Level:
   - Is this the right abstraction for the problem?
   - Are we adding unnecessary abstractions?
   - Does this fit existing patterns (BlockProcessor, metadata, etc.)?

3. Long-term Maintainability:
   - What happens when Demucs updates?
   - What happens when Demucs changes behavior?
   - Can we test this without Demucs installed?

4. Consistency:
   - Does this follow block metadata patterns?
   - Does this follow error handling patterns?
   - Does this follow the "thin wrapper" approach?

CONCERNS:
[List specific architectural concerns]

VOTE: [Approve/Approve w/ Conditions/Reject w/ Alternative/Reject]

REASONING:
[Architectural justification]
```

### Systems Lens: Demucs Proposal Template

```
PROPOSAL: [Name]

SYSTEMS ANALYSIS:

1. Resource Impact:
   - Memory: [Increase/Decrease/Neutral]
   - CPU/GPU: [Increase/Decrease/Neutral]
   - Disk: [Increase/Decrease/Neutral]
   - Network: [Required/Not Required]

2. Failure Modes:
   - New failure modes introduced: [List]
   - Error handling strategy: [Describe]
   - Recovery paths: [Describe]
   - User impact of failures: [Describe]

3. Performance:
   - Expected performance impact: [Quantify if possible]
   - Scalability with audio length: [Linear/Non-linear/N/A]
   - Scalability with concurrent use: [Impact]

4. Stability:
   - Dependency changes: [List]
   - Version compatibility: [Concerns?]
   - Testing requirements: [What's needed]

5. Monitoring:
   - Can we observe what's happening? [Yes/No]
   - Can we debug failures? [Yes/No]
   - What metrics matter? [List]

CONCERNS:
[List specific systems concerns]

VOTE: [Approve/Approve w/ Conditions/Reject w/ Alternative/Reject]

REASONING:
[Systems justification]
```

### UX Lens: Demucs Proposal Template

```
PROPOSAL: [Name]

UX ANALYSIS:

1. User Workflows:
   - What user workflows does this affect? [List]
   - Does it make common tasks easier? [Yes/No/N/A]
   - Does it make complex tasks possible? [Yes/No/N/A]

2. Discoverability:
   - Can users find this feature? [Yes/No]
   - Is it obvious how to use? [Yes/No]
   - Does it need documentation? [Yes/No]

3. Consistency:
   - Consistent with other blocks? [Yes/No]
   - Consistent with command patterns? [Yes/No]
   - Consistent with error patterns? [Yes/No]

4. Cognitive Load:
   - New concepts to learn: [List]
   - Increased complexity: [Low/Medium/High]
   - Can users ignore if not needed? [Yes/No]

5. Error Experience:
   - Are errors clear? [Yes/No]
   - Are errors actionable? [Yes/No]
   - Common pitfalls addressed? [Yes/No]

CONCERNS:
[List specific UX concerns]

VOTE: [Approve/Approve w/ Conditions/Reject w/ Alternative/Reject]

REASONING:
[UX justification]
```

### Pragmatic Lens: Demucs Proposal Template

```
PROPOSAL: [Name]

PRAGMATIC ANALYSIS:

1. Implementation Complexity:
   - Lines of code estimate: [Number]
   - Components affected: [List]
   - Difficulty level: [Low/Medium/High]

2. Testing:
   - Unit tests needed: [List]
   - Integration tests needed: [List]
   - Manual testing required: [List]
   - Requires Demucs installed? [Yes/No]

3. Scope:
   - Can this be broken down? [Yes/No, how?]
   - What's the MVP? [Describe]
   - What can be deferred? [List]

4. Maintenance:
   - Ongoing maintenance burden: [Low/Medium/High]
   - Documentation needed: [List]
   - User support impact: [Low/Medium/High]

5. Risk:
   - What could go wrong? [List]
   - Reversibility: [Easy/Medium/Hard]
   - Cost of being wrong: [Low/Medium/High]

CONCERNS:
[List specific pragmatic concerns]

VOTE: [Approve/Approve w/ Conditions/Reject w/ Alternative/Reject]

REASONING:
[Pragmatic justification]
```

## Common Proposal Patterns & Guidance

### Pattern 1: "Add GUI Feature for Separation"

**Typical Proposals:**
- Visual stem waveform display
- Drag-and-drop model selection
- Real-time separation progress bar
- Stem volume mixing interface

**Council Analysis:**

**Architect:**
-  GUI is separate concern (ApplicationFacade enables)
-  No changes to SeparatorBlock logic
- ️ Consider: Where does UI state live?

**Systems:**
- ️ Loading multiple stems: Memory impact
- ️ Real-time waveform: CPU impact
-  Progress bar: Just parse output, low cost

**UX:**
-  High value (visualization helpful)
-  Aligns with visual workflow builder goal
- ️ Don't over-complicate (simple is better)

**Pragmatic:**
- ️ Medium complexity (Qt UI, audio visualization)
- ️ Test carefully (UI cleanup, memory leaks)
-  Can be iterative (start simple, add features)

**Recommendation Pattern:**
-  APPROVE simple progress/selection UI
- ️ APPROVE WITH CONDITIONS for visualization (memory management)
-  REJECT if requires architectural changes

### Pattern 2: "Optimize Performance"

**Typical Proposals:**
- Cache separation results
- Parallel processing
- Model quantization
- Custom CUDA kernels

**Council Analysis:**

**Architect:**
-  Caching: Good separation of concerns (cache layer)
- ️ Parallel: May affect execution engine design
-  Custom kernels: Deep coupling, violates simplicity

**Systems:**
-  Caching: Reduces redundant work (good)
- ️ Parallel: Resource management complexity
-  Custom kernels: High risk, maintenance burden

**UX:**
-  Faster = better (if transparent)
- ️ Cache invalidation: Must be invisible or clear

**Pragmatic:**
-  Caching: Medium complexity, high value
- ️ Parallel: High complexity, test carefully
-  Custom kernels: Massive scope, defer

**Recommendation Pattern:**
-  APPROVE simple caching (hash-based, clear invalidation)
- ️ APPROVE WITH CONDITIONS parallel (depends on execution engine)
-  REJECT low-level optimizations (trust Demucs)

### Pattern 3: "Add Advanced Configuration"

**Typical Proposals:**
- Expose all Demucs CLI flags
- Fine-tune separation parameters
- Custom model paths
- Advanced audio options

**Council Analysis:**

**Architect:**
-  Metadata approach works (flexible)
- ️ Too many options: Clutters interface

**Systems:**
-  More control: Can optimize for hardware
- ️ More options: More failure modes

**UX:**
-  Complexity explosion: Cognitive overload
- ️ Only add if clear user need

**Pragmatic:**
-  Low implementation cost (metadata)
-  High documentation cost
-  High support cost (users confused)

**Recommendation Pattern:**
-  APPROVE if solves common problem (e.g., two_stems)
- ️ APPROVE WITH CONDITIONS if power user feature (hide by default)
-  REJECT if obscure/rarely needed

### Pattern 4: "Replace Demucs"

**Typical Proposals:**
- Switch to Spleeter, OpenUnmix, etc.
- Use commercial API (Audionamix, iZotope)
- Implement custom separation

**Council Analysis:**

**Architect:**
- ️ Breaking change: All users affected
- ️ Different integration: Code changes
-  High coupling risk

**Systems:**
- ️ New dependency: Stability unknown
- ️ Different performance: Must benchmark
-  Commercial API: Cost, network dependency

**UX:**
-  Breaking change: User confusion
-  Different models: Compatibility issues
- ️ Only if significantly better

**Pragmatic:**
-  High scope: Extensive testing needed
-  Migration path: Complex
-  Risk: High cost if wrong

**Recommendation Pattern:**
-  REJECT unless Demucs is fundamentally broken (it's not)
- ️ CONSIDER ALTERNATIVE: Support multiple backends (high complexity)
-  APPROVE: Improvements to current integration

## Red Flags: Automatic Concerns

When you see these in a Demucs proposal, raise concerns:

### Architectural Red Flags
- "Embed Demucs as library" → Coupling increase
- "Create custom model training pipeline" → Scope explosion
- "Add ML inference in EchoZero" → Wrong layer
- "Manage model downloads ourselves" → Reinventing wheel

### Systems Red Flags
- "Real-time separation" → Not Demucs' design
- "Streaming audio separation" → Architecture mismatch
- "No error handling needed" → Ignoring failure modes
- "Infinite caching" → Resource leak

### UX Red Flags
- "Expose all Demucs options" → Cognitive overload
- "Users can configure everything" → Complexity explosion
- "No defaults needed" → Unusable without config
- "Advanced users only" → Ignoring 80% use case

### Pragmatic Red Flags
- "Quick prototype, we'll fix later" → Technical debt
- "Just add dependency X" → Unevaluated risk
- "No tests needed, it's simple" → Future bugs
- "We might need this someday" → Premature generalization

## Green Flags: Positive Indicators

When you see these, evaluate favorably:

### Architectural Green Flags
- Maintains CLI wrapper approach
- Respects existing boundaries
- Fits block metadata pattern
- No new abstractions needed

### Systems Green Flags
- Improves error handling
- Adds observability
- Reduces resource usage
- Simplifies deployment

### UX Green Flags
- Makes common case simpler
- Improves error messages
- Better progress visibility
- Maintains consistency

### Pragmatic Green Flags
- Small, focused scope
- Easy to test
- Clear success criteria
- Reversible decision

## Decision Checklist

Before voting, verify:

- [ ] Does this align with "best part is no part"?
- [ ] Does this align with "simplicity and refinement"?
- [ ] Is the problem real (evidence provided)?
- [ ] Is this the simplest solution?
- [ ] Does this respect Demucs boundaries?
- [ ] Are failure modes considered?
- [ ] Is error handling adequate?
- [ ] Is user impact positive?
- [ ] Is complexity cost justified?
- [ ] Can this be tested effectively?
- [ ] Is maintenance burden acceptable?
- [ ] Would I be happy debugging this at 3am?

## Voting Examples

### Example 1: "Add MPS Support When Available"

**Architect:**  APPROVE - Just update detect_best_device(), no architecture change
**Systems:**  APPROVE - Free 10x speedup on Apple Silicon when PyTorch ready
**UX:**  APPROVE - Invisible to users, just faster
**Pragmatic:**  APPROVE - 5 lines of code, conditional on PyTorch support

**Council Recommendation:** PROCEED - Monitor PyTorch issue, implement when stable

### Example 2: "Add Stem Selection for Downstream Blocks"

**Architect:**  APPROVE - Metadata-based selection, fits pattern
**Systems:**  APPROVE - No resource impact, just selects from list
**UX:**  APPROVE - Users want to process specific stems
**Pragmatic:**  APPROVE - Low complexity, clear value

**Council Recommendation:** PROCEED - Add stem_index or stem_name metadata field

### Example 3: "Implement Custom Vocal Separation Algorithm"

**Architect:**  REJECT - Violates separation of concerns, huge scope
**Systems:**  REJECT - Unproven stability, maintenance nightmare
**UX:** ️ NEUTRAL - Only better if quality exceeds Demucs (unlikely)
**Pragmatic:**  REJECT - Massive scope, low chance of success

**Council Recommendation:** REJECT - Demucs is state-of-the-art, focus on integration quality

## Summary

When evaluating Demucs proposals:

1. **Maintain Boundaries:** Thin CLI wrapper is correct approach
2. **Respect Complexity:** ML separation is complex, defer to experts
3. **User Value First:** Does this solve real problems?
4. **Simplicity Bias:** Default to rejection, require justification
5. **Integration Quality:** Focus on making current approach excellent

Remember: Demucs integration is a success story for EchoZero's values. Most proposals should refine, not reimagine.

**The best part is no part. The best improvement is removing complexity while adding value.**


