# Project Adaptive Review Loop Spec

Status: active
Last reviewed: 2026-04-30



This spec defines the EZ-owned operator loop that turns explicit Project review
into better runtime bundles.

Use this document when the question is:
- what the adaptive review loop must do end to end
- which surface owns review, dataset, training, and runtime adoption
- what a promoted Project-specialized model is allowed to change
- how today's implementation slices fit the broader milestone

Use [STATUS.md](STATUS.md) for repo truth.
Use [REVIEW-SIGNAL-FEATURE-SPEC.md](REVIEW-SIGNAL-FEATURE-SPEC.md) for the
shared review-signal contract.
Use [FOUNDRY-TRAINING.md](FOUNDRY-TRAINING.md) for current Foundry training
and artifact behavior.
Use [PROJECT-ADAPTIVE-REVIEW-LOOP-EXECUTION-PLAN.md](PROJECT-ADAPTIVE-REVIEW-LOOP-EXECUTION-PLAN.md)
for slice order and proof planning.

## Why This Exists

EchoZero already had most of the pieces for a useful model-improvement loop:
- real Project import and batch analysis
- phone review and timeline correction surfaces
- durable review signals in Foundry
- dataset and train-run infrastructure in Foundry
- installed runtime bundles for future classification work

What it did not have was one stable product contract that connects those pieces
into one operator loop with clear ownership boundaries.

Without that contract:
- review can drift into one-off sidecar workflows
- Project datasets become hard to find or reuse safely
- promoted model outputs stay detached from future pending work
- training and runtime adoption leak into direct truth mutation

This spec closes that gap.

## Product Goal

Create one canonical loop where the operator can:
1. import a Setlist into a Project
2. analyze a bounded seed batch with the current installed runtime bundles
3. review or correct detected Events in EZ or phone review
4. commit explicit missed Events from EZ fix mode
5. persist all explicit review into one canonical review-signal lane
6. inspect the resulting Project review dataset from EZ
7. create a Project-specialized model from that persisted dataset
8. validate, install, and promote the resulting runtime bundle
9. apply the promoted bundle to pending work without rewriting reviewed truth
10. continue with the next batch

## Locked Rules

- Main remains truth.
- Takes remain subordinate history and candidate lanes.
- Review signal is explicit. Untouched Events are not positive signal.
- Timeline review writes through the canonical application path, never through
  widget-local persistence.
- Foundry is the persisted review, dataset, training, and artifact lane.
- EZ is the primary operator workflow lane for this milestone.
- Project-specialized training resolves from persisted dataset versions, not
  raw review rows or temp folder guesses.
- Model promotion and model adoption are separate actions.
- Adopting a promoted model must not silently rewrite already reviewed or
  already processed main truth.
- Custom pinned model paths stay pinned unless the operator explicitly changes
  them.

## Canonical Lanes

### 1. Review Signal Lane

One durable record for one explicit human review decision.

This lane owns:
- decision vocabulary
- provenance
- project writeback hooks
- dataset-materialization eligibility

This lane does not own:
- queue navigation semantics
- dataset versioning
- training runs
- runtime bundle selection

### 2. Project Review-Dataset Lane

Versioned, discoverable Project-scoped datasets derived from explicit review
signal.

This lane exists so the operator can inspect what one Project has taught us so
far without mixing that data into global training by accident.

### 3. Shared Core-Dataset Promotion Lane

One explicit promotion target for reviewed data that is accepted for reuse
across Projects.

This lane is operator-visible, lineage-backed, and deduplicated where possible.
It is never an automatic spillover from every Project review dataset.

### 4. Model Build Lane

A bounded train-run lane that consumes one chosen dataset version and produces
artifacts plus validation output.

For the first implementation slice, the Project-specialized recipe may stay
kick/snare focused, but the dataset and review lanes must remain general.

### 5. Runtime Bundle Lane

The lane that validates, installs, resolves, and adopts promoted bundles for
future pending work.

This lane never mutates review signal or dataset truth.

## Ownership Boundary

- EZ app shell and timeline application:
  operator workflow, explicit review actions, Project launch surfaces, pending
  adoption policy
- Foundry services:
  persisted review records, review-derived dataset versions, train runs,
  artifact validation, bundle installation, dataset queries
- Timeline widgets:
  input surfaces only

## Operator Contract

The operator-facing contract for this milestone is:

- review creates signal
- signal creates Project review datasets
- Project review datasets can train a Project-specialized model
- Project review datasets can also promote into a shared core dataset
- promoted bundles can become the default for pending work
- none of those actions silently rewrite already reviewed truth

## Project-Specialized Model Contract

`Create Specialized Model` means:
- resolve the latest or selected persisted Project review-dataset version
- derive the bounded train input from that dataset version
- create and run the Foundry training job
- validate the resulting artifact
- install the resulting runtime bundle
- return enough references for EZ to report what happened

It does not mean:
- mutate timeline truth
- backfill untouched review decisions
- overwrite every saved model path in the Project
- silently adopt the new bundle for already processed Songs

For the current v1 slice:
- the first Project-specialized build path may remain binary drum focused
- promotion should be rollback-safe if validation or install fails partway
- EZ may refresh pending default configs immediately after successful promotion
  when those configs still follow the previous global defaults or are blank

## Pending-Work Adoption Rules

Adopting a promoted model is explicit and scoped.

V1 adoption must:
- target pending or unprocessed work
- preserve reviewed or already processed main truth
- leave custom pinned model paths untouched
- make the promoted bundle easy to use for the next batch

V1 adoption may start with a conservative rule:
- rewrite only blank settings or settings that still point at the immediately
  previous installed defaults

Broader scoped adoption such as active Song, selected Songs, or remaining
Setlist work can build on that rule later.

## Today's Implementation Slice

The current slice that must become testable first is:
- explicit `missed_event_added` review commit from EZ fix mode
- persisted review-signal write path that is not session-item only
- Project review-dataset discovery from EZ
- Project-specialized model creation from the latest Project review dataset
- conservative pending-config refresh after promotion

This slice is intentionally smaller than the full milestone, but it must obey
the same ownership boundaries and promotion rules.

## Acceptance Criteria

This milestone is on-contract when all of the following are true:

- explicit review from phone review and EZ can write the canonical review lane
- manually added missed Events can become durable reusable positive signal
- Project review datasets are durable and discoverable from EZ
- EZ can create a Project-specialized model from persisted Project review data
- successful promotion validates and installs a runtime bundle
- pending work can adopt the promoted bundle without hand-editing every Layer
- already reviewed truth remains stable unless the operator explicitly reruns it

## Not In This Spec's Done Bar

Not required for this milestone:
- continuous online self-training
- silent approval from untouched Events
- automatic global dataset promotion from every Project
- automatic global bundle switching with no operator confirmation
- full hyperparameter editing in the first EZ specialized-model action
