# Canonical Review Foundation Plan

Status: active implementation plan  
Last updated: 2026-04-26

## Goal

Replace the current split review/session/item model with one cleaner foundation:

1. canonical event truth lives in the application timeline
2. phone review and timeline fix mode are two surfaces over the same event model
3. song and project training datasets are derived artifacts, not editing truth

This plan is the repo-ready execution contract for that collapse.

## Scope Note

- This document is the active implementation plan for the canonical review
  foundation.
- `docs/PROJECT-ADAPTIVE-REVIEW-LOOP-SPEC.md` remains the broader product
  contract for project review and adaptive model improvement.
- `docs/REVIEW-SIGNAL-FEATURE-SPEC.md` remains the current Foundry review-signal
  contract.
- `docs/FOUNDRY-TRAINING.md` remains the training/export reference for Foundry.

## Problem

The current backend shape is carrying too many truth-like objects:

- timeline/project event truth
- review-session state
- review-item state
- review-signal state
- dataset materialization state

That creates soft boundaries and contradictory behavior:

- queue/session objects feel more canonical than they should
- timeline review and phone review are not operating on the same live truth
- dataset creation is too close to operator edits
- rejected or timeline-only fixes can miss the canonical downstream lanes

The product direction is simpler:

- persist every detected onset
- classify promoted versus demoted instead of keep versus drop
- let operators directly correct canonical events
- derive training artifacts later from explicit confirmed state

## Locked Decisions

- Main event truth stays in the application timeline.
- Every detected onset persists as an event.
- Classification threshold decides `promoted` versus `demoted`, not existence.
- Demoted events remain visible, queryable, and editable.
- Demoted events must not drive normal playback, export, sync, or MA3-facing
  truth.
- Manually added missed events are created as promoted events.
- Relabel and retime update the canonical event and preserve correction lineage.
- Unchanged promoted events become sign-off candidates only when the operator
  explicitly closes a review/fix scope and confirms submission.
- There is no durable `ProjectReviewItem` truth object in the target model.
- Review sessions are ephemeral scopes or saved filters, not product truth.
- Song and project review datasets are derived from canonical event state plus
  explicit review metadata.
- Foundry consumes exported dataset versions, not live editing state.

## Target Event Model

Canonical timeline events should carry:

- timing and display fields already used by the app
- `origin`
  - examples: `model_detected`, `manual_added`, `ma3_pull`
- full `classifications`
- full `metadata`
- typed review semantics derived from metadata
  - `promotion_state`: `promoted` or `demoted`
  - `review_state`: `unreviewed`, `corrected`, or `signed_off`

Required metadata envelopes:

- `review`
  - `schema`
  - `promotion_state`
  - `review_state`
  - later slices may add correction lineage fields here
- `detection`
  - `schema`
  - classifier score
  - positive threshold
  - threshold pass result
  - model/source provenance

Important rule:

- the runtime event must preserve full domain classifications and metadata
  through app projection, edit, duplicate, move, and persistence paths

## Slice 1
Canonical Event Foundation

Goal:
- make the canonical app event capable of holding review and detection truth

Scope:
- extend timeline event runtime shape to preserve `origin`, `classifications`,
  and `metadata`
- derive `promotion_state` and `review_state` from metadata on the canonical
  event object
- preserve these fields through project-take projection and save round-trips
- update edit/duplicate/clone paths so metadata is not lost
- change binary drum classification so all candidates persist
- stamp classification outputs with explicit promoted or demoted review metadata
- surface demoted and signed-off state in the timeline presentation layer
- exclude demoted event-slice playback from the normal playback render path

Likely files:

- `echozero/application/timeline/models.py`
- `echozero/application/timeline/assembler_layers.py`
- `echozero/application/timeline/assembler_signature.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/orchestrator_event_edit_mixin.py`
- `echozero/application/playback/runtime.py`
- `echozero/ui/qt/app_shell_project_timeline_storage.py`
- `echozero/ui/qt/app_shell_layer_storage.py`
- `echozero/processors/binary_drum_classify.py`
- classified-drum pipeline template docs

Done when:

- a classified onset below threshold still persists as a demoted event
- a classified onset above threshold persists as a promoted event
- classified review metadata survives app projection and save round-trips
- duplicate, move, and take-promotion flows preserve metadata
- demoted event-slice playback is excluded from the rendered playback buffer

Proof:

- focused processor tests for promoted and demoted classification candidates
- focused storage round-trip tests for review and detection metadata
- app/timeline presentation proof that demoted state surfaces through the real
  event projection path

## Slice 2
Shared Review Query And Command Layer

Goal:
- remove separate phone/timeline truth models and move both surfaces onto one
  application review layer

Scope:

- add one query model for reviewable events
- add one command set
  - promote
  - demote
  - add missed
  - relabel
  - retime
  - finalize review scope
- treat review sessions as ephemeral scopes with cursor state and closeout
  confirmation only
- stop depending on persisted review-item truth for operator interaction

Likely files:

- `echozero/application/timeline/review/*`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/ui/qt/app_shell_timeline_review.py`
- `echozero/foundry/review_server.py`
- `echozero/foundry/review_web.py`

Done when:

- phone review and timeline fix mode read the same event-backed review surface
- both paths issue the same canonical commands
- untouched promoted events only become sign-off on confirmed review-scope
  closeout

## Slice 3
Derived Dataset And Foundry Handoff

Goal:
- make training/export a derived lane from canonical event truth

Scope:

- build one application-side project review corpus from canonical events plus
  review metadata
- expose song-scoped filtered views over that corpus
- materialize versioned song and project datasets only on explicit
  export/train/publish
- point specialized-model flows at exported dataset versions, not sessions

Likely files:

- `echozero/foundry/services/dataset_service.py`
- `echozero/foundry/services/query_service.py`
- `echozero/foundry/services/project_specialized_model_service.py`
- `echozero/ui/qt/app_shell_project_review.py`

Done when:

- project dataset creation no longer depends on review-session truth
- song/project dataset versions are explicit exported artifacts
- Foundry training uses exported dataset versions with lineage

## Guardrails

- Do not reintroduce a durable `ProjectReviewItem` truth object.
- Do not make review-session state authoritative.
- Do not drop below-threshold classified events.
- Do not let demoted events drive normal playback or sync behavior.
- Do not mutate datasets directly on every operator click.
- Do not bypass the canonical application boundary from timeline widgets.

## Current Slice 1 Status

Started on 2026-04-26:

- canonical runtime events now preserve full domain classifications and metadata
- classified drum outputs are being moved from keep/drop semantics to
  promoted/demoted persistence
- project-take round-trip coverage is being tightened so review metadata cannot
  disappear during normal app edits
