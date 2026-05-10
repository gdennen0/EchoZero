# Project Adaptive Review And Training Plan

Status: active
Last reviewed: 2026-05-02



## Goal

Build one product-grade EchoZero training system with two linked outcomes:

1. local Project review should produce trustworthy training signal and safe
   Project-specialized local models
2. consented reviewed data from many users should be promotable into large
   shared EchoZero training datasets for better base models

This is the canonical planning doc for that broader training direction.

## Scope Note

- This document is the active broad plan for adaptive review, training,
  distribution, and consented data contribution.
- `docs/REVIEW-SIGNAL-FEATURE-SPEC.md` remains the shared review-signal
  contract.
- `docs/REVIEW-SIGNAL-EXECUTION-PLAN.md` remains the narrower dependency plan
  for review-signal implementation slices.
- `docs/FOUNDRY-TRAINING.md` remains the current training/export/validation
  reference for Foundry surfaces.
- `docs/APP-DELIVERY-PLAN.md` remains the canonical packaging and app-first
  delivery plan for the desktop app itself.

## Product Thesis

EchoZero should not treat model training as a sidecar lab workflow.

The target shape is:

- users run the real app locally
- operators review and correct real Event truth through the app and phone lanes
- explicit review becomes durable training signal
- that signal can build local Project-specialized models on the user machine
- users may optionally consent to contribute reviewed samples back to EchoZero
- consented reviewed data can improve the shared base models over time

Simple rule:

- local review improves local work first
- explicit consent is required before reviewed data leaves the user machine
- shared base-model growth is a separate promotion lane, never an accidental
  side effect of local use

## Locked Constraints

- Main remains truth.
- Takes remain subordinate history and candidate lanes.
- Only explicit review commits become training signal.
- Timeline review writes through the application boundary, never directly from
  widgets.
- Foundry owns training, artifact validation, and model-build plumbing.
- Project-local training must work without requiring cloud services.
- Shared-dataset promotion must be explicit, lineage-backed, and revocable at
  the policy level.
- User data must not leave the machine without a clear consent contract and
  operator-visible scope.
- Local model promotion and shared dataset promotion are separate product
  actions with separate UI, storage, and audit semantics.

## Two Linked Lanes

### Lane A
Local Product Scope

Goal:
- let one user review one Project, train one bounded local model, validate it,
  and use it safely for pending work

Outcome:
- better Project-specific performance without waiting on central model updates

### Lane B
Platform Distribution And Consented Contribution

Goal:
- distribute EchoZero as a user-facing app, support updates and packaging, and
  let users explicitly contribute reviewed samples for shared base-model
  improvement

Outcome:
- broader installed base, durable reviewed corpora, and stronger base models

The product is not complete unless both lanes are planned explicitly, but the
execution order should prioritize Lane A first.

## Local Product Scope

The local v1 bar is:

1. import a Setlist into a real EZ Project
2. run extraction and classification with installed runtime bundles
3. review detected Events in EZ or phone review
4. verify, reject, relabel, retime, and add missed Events through canonical
   review paths
5. persist those explicit review actions into one durable review-signal lane
6. materialize Project review datasets from that signal
7. train a Project-specialized model locally through Foundry
8. validate, install, and promote the local model safely
9. let the operator apply the promoted model to pending Songs or Layers
10. keep prior reviewed truth intact while new pending work benefits from the
    better model

### Local V1 Ship Boundary

Must exist:

- one canonical review signal written by both EZ and phone review
- explicit support for verified, rejected, relabeled, boundary-corrected, and
  missed-event-added decisions
- durable Project review dataset creation with lineage
- one bounded local training entrypoint
- compatibility validation before runtime promotion
- logical local model selection for pending work
- visible rollback or revert path for local bundle adoption

Must not be required for local v1:

- cloud accounts
- automatic upload
- central training infrastructure
- cross-user sharing
- background silent promotion into the shared base-model corpus

### Local V1 Non-Goals

- continuous online self-training
- implicit approval from untouched Events
- silent replacement of already-reviewed truth
- requiring Foundry desktop UI as the primary operator surface

## Platform Distribution And Consented Contribution

The long-term platform bar is:

1. users can install and update EchoZero through a clear supported packaging
   path
2. users can review and train locally without being forced into data sharing
3. users can opt into sample contribution with explicit consent
4. contributed data lands in a governed intake lane, not directly in base-model
   training
5. shared datasets, training runs, validation, and promotion into base models
   remain curated and lineage-backed

### Distribution Questions This Plan Must Answer

- how the app is packaged and updated for end users
- how licensing, accounts, or activation interact with local model features
- how local model files, reviewed datasets, and runtime bundles are stored
- how support, diagnostics, and version compatibility are handled
- how release cadence affects model compatibility and migration

### Consent And Data Contribution Questions This Plan Must Answer

- what the user is consenting to share
- whether shared payloads include raw audio, clipped Events, labels, review
  decisions, model provenance, or only derived slices
- how sensitive metadata is minimized or removed
- how contribution can be toggled on or off
- whether consent is Project-wide, workspace-wide, or item-by-item
- how contributed data is queued, retried, audited, and withdrawn from future
  corpus builds
- what legal and product language governs ownership, use, and revocation

### Platform Lane Principles

- local use must remain useful without contribution
- contribution must be opt-in, not default
- upload payload shape must be narrower than local persistence shape
- intake and curation must be explicit before base-model training use
- global base-model promotion must have stronger gates than local Project-model
  promotion

## Shared Data Model

The broader system should settle into five distinct lanes:

1. review signal lane:
   one canonical durable record of explicit human review from EZ or phone
   review
2. Project dataset lane:
   versioned Project-scoped datasets built from review signals for local
   training
3. consented contribution lane:
   explicit export or upload packages approved by the user for external sharing
4. shared core-dataset lane:
   curated cross-user reviewed corpora derived from accepted contributed
   packages
5. model build and runtime bundle lanes:
   validated training, promotion, install, selection, and rollback flows for
   both local specialized models and shared base models

Simple rule:

- review creates signal
- signal creates Project datasets
- Project datasets can train local Project models
- consented exports can feed curated shared datasets
- curated shared datasets can train future base models

## Current Baseline

Implemented now:

- Foundry has real dataset, run, export, and validation surfaces
- local sample-library retraining exists
- Project-backed phone review exists from the EZ shell
- explicit review signals already exist in Foundry
- Project-backed review can already write corrected truth back into Project
  main-take data when provenance is sufficient
- reviewed records can already materialize into dataset-ready samples
- a Project-specialized drum-model orchestration path already exists
- runtime bundle install and installed-bundle resolution already exist

Partially implemented now:

- timeline fix mode still needs to become a first-class producer of explicit
  review commits
- the broader operator flow from reviewed data to promoted local model is not
  yet one clean EZ-owned surface
- shared-dataset promotion and consented export are not yet productized
- logical pending-work model adoption semantics are not yet fully settled

Missing now:

- one clean canonical local training workflow from EZ review to local promoted
  model
- consent UX and policy-backed export semantics
- explicit contributed-sample package format
- shared dataset intake, curation, and promotion lane
- base-model retraining and distribution plan for contributed reviewed data
- operator-visible separation between local model promotion and shared data
  contribution

## Execution Order

### Phase 0
Stabilize Review-Signal And Local Training Foundations

Goal:
- finish the core local review and Foundry plumbing before adding any
  distribution or upload features

Includes:

- canonical review commit boundary
- timeline fix-mode write-through
- missed-event and boundary-correction capture
- durable Project dataset extraction
- local specialized-model creation, validation, install, and pending-work
  adoption

Done when:

- one operator can complete the full local loop through the real app path
- no cloud service is required
- review and model-promotion proofs are app-boundary correct

### Phase 1
Productize The Local Operator Loop

Goal:
- make the local loop understandable, discoverable, and reversible for a real
  end user

Includes:

- clear EZ-first UI for review progress, collected dataset state, train status,
  validation status, and model adoption
- safe rollback and status surfaces for local model changes
- packaging and smoke consideration for local-model workflows
- operator-facing docs and support surfaces

Done when:

- a real user can review, train, validate, promote, adopt, and roll back a
  Project-local model without using sidecar dev workflows

### Phase 2
Define Contribution Contract And Export Package

Goal:
- create the explicit legal, product, and technical contract for sharing
  reviewed samples

Includes:

- consent UX rules
- contribution scope rules
- export package schema
- metadata minimization rules
- upload audit and retry behavior
- revocation and policy boundaries

Done when:

- a user can clearly understand what they are sharing
- the product has one explicit package shape for contributed reviewed data
- no ambiguous silent-upload behavior remains

### Phase 3
Build Shared Intake And Curation Lane

Goal:
- receive consented contribution packages and convert them into curated shared
  training corpora

Includes:

- contribution intake storage
- deduplication and lineage
- curation and rejection flow
- shared dataset versioning
- metrics and audit surfaces

Done when:

- contributed reviewed data can be accepted or rejected explicitly
- shared datasets are inspectable, reproducible, and governance-backed

### Phase 4
Train And Distribute Better Base Models

Goal:
- use curated shared datasets to train stronger base models and deliver them
  back to users safely

Includes:

- shared base-model training recipes
- validation and regression gates stronger than local Project-model gates
- release packaging for new base models
- compatibility and migration rules for deployed user installs

Done when:

- better base models can be trained from curated shared corpora
- those models can be shipped back to users without breaking local workflows

## Decision Stack

The next planning and implementation work should answer these questions in
order:

1. what exact local operator flow do we want to ship first
2. what exact local model selection and rollback semantics apply to pending
   Songs and Layers
3. what exact payload can leave the machine under consent
4. what exact governance path promotes contributed data into shared corpora
5. what exact release path distributes improved base models back to users

If this order is inverted, the project will drift into premature platform work
before the local product loop is trustworthy.

## Immediate Next Planning Outputs

Write next:

1. a local operator contract doc for EZ-side review, train, validate, promote,
   adopt, and rollback behavior
2. a consent and contributed-sample package spec
3. a shared dataset intake and curation spec
4. a base-model promotion and distribution spec

## Guardrails

- Do not make cloud upload part of the local v1 completion bar.
- Do not make consent implicit or bundled into unrelated app actions.
- Do not let Project data silently become shared training data.
- Do not collapse local model promotion and shared-data contribution into one
  button.
- Do not bypass Foundry validation for either local or shared model promotion.
- Do not let contributed-data governance rewrite canonical Project truth.
