# Model Evolution Architecture

Status: draft
Last verified: 2026-05-20
Lane: Foundry

## Purpose

Model evolution should become a first-class EchoZero capability: the user fixes Events in the real timeline, then EchoZero can improve the relevant models from those fixes with a calm, simple path.

The target experience is not a training console. It is a product principle: every correction can become future intelligence.

## Principle

Fixed Events are user-confirmed truth. Model training should consume that truth through a dedicated Foundry path that creates runtime-shaped examples, continues from the current best model when appropriate, validates behavior through the app-facing model contract, and installs only clearly named runtime bundles.

The user-facing version should feel small:

- Improve current models.
- Create new model.
- Preview before install.
- Roll back.

Everything else is an advanced detail.

## Lesson From The Noah Kahan Experiment

The Noah Kahan kick/snare run proved the mechanics can work: shared review samples can be ingested, one-vs-rest datasets can be built, models can warm-start from existing bundles, and runtime bundles can be installed under a new identity.

It also exposed a deeper issue:

- The shared review sample pool appears to contain mostly 80 ms clips.
- Those clips likely mirror timeline Event duration, not full training-context duration.
- Runtime inference evaluates a larger audio window, so padded 80 ms clips can become a poor training distribution.
- A continuously evolving core model needs both user truth and runtime-shaped training material.

The fix is not just “train harder.” The system needs a dedicated model evolution module that separates Event truth from model-ready audio windows.

## Terms

**Fixed Event** means a timeline Event accepted or corrected by the user.

**Review Truth** means the durable record of the user decision: source project, song, source audio, time range, class, decision kind, and confidence/context metadata.

**Runtime-shaped Sample** means an audio example materialized with the same temporal context and preprocessing shape the runtime model sees during inference.

**Model Lineage** means the relationship between a new candidate model and the installed model or checkpoint it continued from.

**Candidate Model** means a trained model that has not yet passed promotion gates.

**Installed Model** means a runtime bundle available to the app model selector.

## Proposed Module

Create a dedicated Foundry module:

```text
echozero/foundry/model_evolution/
  __init__.py
  truth_collector.py
  sample_materializer.py
  lineage.py
  planner.py
  trainer.py
  promotion.py
  install.py
  service.py
```

The existing lower-level services should remain useful:

- `DatasetService` owns dataset versions and deterministic sample identity.
- CRNN trainer services own artifact training and warm-start loading.
- Runtime bundle install services own bundle compatibility and app registration.
- Project-specialized services remain intact and should not be silently redirected.

The new module coordinates them into one clear product path.

## Internal Flow

```text
Fixed Events
  -> Review Truth Collector
  -> Runtime Window Materializer
  -> Dataset Version
  -> One-vs-Rest Dataset Builder
  -> Lineage Resolver
  -> Evolution Run Planner
  -> Trainer
  -> Evaluation + Promotion Gate
  -> Runtime Bundle Install
  -> App Model Selector
```

### Review Truth Collector

Collect all eligible fixed Events from the selected scope:

- current song
- current project
- shared local review pool
- organization-wide library later

The collector should preserve Event timing and decision metadata. It should not assume the Event visual duration is the right model sample duration.

### Runtime Window Materializer

Materialize model-ready samples from source audio using the model family’s expected window policy.

For drum binary models, this likely means a deterministic window around the Event onset or corrected anchor, with the same duration and preprocessing expected by runtime classification.

The output should record:

- stable sample id
- content hash
- source audio path or source id
- Event start/end
- materialized window start/end
- class
- source kind
- preprocessing/window policy version

### Dataset Builder

Build one-vs-rest datasets per target class.

For `kick`, positives are fixed kick examples and negatives are all eligible non-kick classes.

For `snare`, positives are fixed snare examples and negatives are all eligible non-snare classes.

Negatives should be explicit and complete for the chosen scope. The run summary should show the class counts used, including every negative source class.

### Lineage Resolver

Resolve the starting model:

- Improve current model: start from the installed bundle for the same family and class.
- Create new artist/project model: start from the current core model or user-selected lineage seed.
- Continue candidate: start from a prior candidate checkpoint.

Lineage must be written into the run metadata and installed manifest. The app should be able to answer “what did this model come from?”

### Evolution Run Planner

Create an explicit run profile. For serious user-initiated improvement, use a bounded strong profile rather than tiny defaults.

Example profile names:

- `quick_check`
- `beefy`
- `release_candidate`

The main UX should not expose raw epochs, batch size, optimizer, or augmentation. Advanced details can be inspected after the fact.

### Promotion Gate

A candidate should not become the default just because training completed.

Promotion should check:

- validation metrics
- per-class confusion
- threshold calibration
- regression replay against known songs
- app-runtime bundle compatibility
- optional human preview on a small before/after set

For evolving core models, include a replay dataset from prior accepted material to reduce catastrophic forgetting.

## UX Shape

The user-facing surface should be an “Improve Models” experience, not a data science dashboard.

First screen:

- current installed models
- fixed Events available
- newest candidate status
- one primary action: `Improve Models`
- secondary action: `Create New Model`

During training:

- plain progress stages: Preparing examples, Training, Checking results, Ready to review
- no raw logs by default
- a small disclosure for technical details

Review screen:

- current model vs candidate
- changed classifications on a small song/sample set
- positive/negative counts
- lineage seed
- install, keep training, or discard

Model identity should be clear in every installed bundle:

- family
- class
- identity, such as `Noah Kahan`
- lineage seed
- profile
- creation date
- promotion status

## App Contract

The app should treat installed model bundles as product objects, not loose files.

Every installed bundle manifest should include:

- model family
- target class
- display identity
- technical model id
- lineage source manifest/artifact id
- training source scope
- training profile
- threshold policy
- compatibility version
- install status

The drum model selector should only show compatible installed bundles.

## Data Rules

- Generated weights, review WAVs, caches, and local artifacts stay out of git.
- Class folders can be an ingestion source, but source audio plus manifest timing is preferred when regenerating runtime-shaped samples.
- Class folders are source of truth for class membership only when no richer review truth store exists.
- Event truth and materialized audio samples should be versioned separately.
- Deterministic ordering and stable sample ids are required.

## Compatibility With Existing Paths

The project-specialized flow should remain available and unchanged unless explicitly migrated.

The new model evolution path should reuse proven components:

- dataset version creation
- one-vs-rest planning
- warm-start capable CRNN training
- artifact validation
- runtime bundle install/index

It should not hide behavior by changing global defaults. Stronger training should live behind an explicit profile.

## First Implementation Slice

1. Finish the 80 ms investigation and decide whether review samples need regeneration from source audio.
2. Add a model evolution service that can create runtime-shaped drum samples from fixed Events or review truth.
3. Add lineage metadata to candidate runs and installed bundle manifests where missing.
4. Add one app-facing command for `Improve Models` using the current installed class models as seeds.
5. Add a review/promotion gate before setting a candidate as the active default.
6. Add a compact UX surface after the service contract is stable.

## Focused Tests

Add tests for:

- fixed Event truth collection preserves Event timing
- materializer writes runtime-shaped windows independent of visual Event duration
- one-vs-rest datasets include all negatives for the selected scope
- lineage resolver chooses the installed current model as the seed
- installed manifest includes identity, class, profile, and lineage
- promotion gate refuses incompatible or metric-regressing bundles

Tests should fake training where possible. Heavy model training belongs in manual verification commands or dedicated release gates.

## Open Questions

- Should “fixed Event” store a corrected onset anchor separately from visual start/end?
- Should model evolution consume project-local truth, shared-local truth, or both by default?
- What replay set is required before a candidate can update a core model?
- How should thresholds be recalibrated when a model is continued from an existing checkpoint?
- Should the first UX surface live inside Foundry, timeline review, or a small model center reachable from both?

