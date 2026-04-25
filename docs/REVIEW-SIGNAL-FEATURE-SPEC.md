# Review Signal Feature Spec

Status: proposed implementation target
Last verified: 2026-04-24

This spec defines the canonical human-review and training-signal lane for
EchoZero.

Use this document when the question is:
- what review data should exist
- which surfaces are allowed to write it
- what counts as a positive or negative training signal
- how phone review and timeline fix mode stay aligned

Use [STATUS.md](STATUS.md) for repo truth.
Use [../echozero/application/timeline/README.md](../echozero/application/timeline/README.md)
for the canonical timeline application boundary.
Use [../echozero/foundry/README.md](../echozero/foundry/README.md) for Foundry
lane ownership.

## Why This Exists

EchoZero already has the ingredients for a useful feedback loop:
- model-generated events in the timeline
- operator correction in fix mode
- a phone-first review lane in Foundry
- training and artifact flows in Foundry

What it does not yet have is one canonical review signal that both operator
surfaces write into.

Without that contract:
- timeline edits become hard to reuse safely for training
- phone review becomes a sidecar instead of product truth
- reviewed positives and negatives drift into inconsistent shapes
- later project-adaptive models would train on weak or ambiguous signals

This feature closes that gap first.

## Current Baseline

As of 2026-04-24, the repo already has a seeded review lane under Foundry:
- `echozero/foundry/domain/review.py` defines `ReviewItem`,
  `ReviewOutcome`, and the nested `ReviewDecision` shape
- `echozero/foundry/services/review_session_service.py` persists review
  sessions and emits phone-review snapshots
- `echozero/foundry/review_server.py` serves the phone review API and audio
- `echozero/foundry/review_web.py` renders the one-page phone UI
- `tests/foundry/test_review_sessions.py` covers review persistence, API, and
  navigation

The current landed decision shape supports:
- `verified`
- `rejected`
- `relabeled`

The current phone page supports:
- explicit `Prev` and `Next`
- replaying the current clip
- verify, reject, and relabel flows
- deterministic navigation inside the active filtered queue

What is still missing:
- project/song/layer queue generation from the canonical EZ app path
- timeline fix mode write-through into the same persisted review lane
- missed-event and boundary-correction capture
- downstream promotion of reviewed signals into Foundry datasets and runs

## Product Goal

Create one shared review signal lane that:
- treats phone review and timeline fix mode as equal first-class producers
- records explicit human review decisions with durable provenance
- only promotes explicit reviewed signals into training inputs
- stays app-boundary correct for timeline work and service-boundary correct for
  Foundry work
- makes later project-scoped model adaptation safe instead of speculative

## Locked Rules

- Main remains truth. Review data never creates alternate live truth.
- Timeline fix mode must write through the canonical timeline application path,
  not widget-local persistence.
- Foundry remains the training and persisted review lane.
- The phone app and the timeline must emit the same decision vocabulary.
- Untouched events are not positives.
- Only explicit verified positives become positive training signal.
- False positives become explicit negative signal only when the operator
  rejects them.
- Reclassifications and corrected boundaries must preserve both the original
  prediction context and the reviewed correction.
- Queue/session state is work management, not product truth.
- Project-adaptive model training is downstream of this feature, not part of
  its v1 completion bar.

## Scope

This feature covers:
- review queue creation for explicit operator review work
- persisted review records with reusable decision semantics
- phone-first queue review for song/layer work
- timeline fix-mode emission of review records during manual correction
- promotion of reviewed records into Foundry-ready dataset signals

This feature does not cover:
- continuous online self-training
- silent auto-approval of untouched events
- full phone-side event-boundary editing in v1
- project-specific fine-tune scheduling, promotion, or runtime selection

## Canonical Vocabulary

Use these terms in code, docs, and UI copy for this feature:

- Review signal:
  one persisted human-reviewed record that can later feed training or audit
- Review queue:
  a bounded ordered worklist of review targets
- Verified positive:
  an event or clip explicitly approved by a human as correct
- Rejected event:
  a false positive explicitly marked wrong
- Relabeled event:
  an event that exists, but whose classification was corrected
- Boundary-corrected event:
  an event that exists, but whose timing or size was corrected
- Missed event:
  a false negative recovered by the operator through manual creation

## Canonical Review Contract

The seeded Foundry `ReviewDecision` shape is the starting point, not the final
ceiling.

The canonical review signal must capture:
- source identity:
  project, song, version, layer, event, and queue/session references when they
  exist
- prediction snapshot:
  predicted classification, score/confidence, model/bundle identity, and any
  source event linkage
- review decision:
  what the human concluded
- correction payload:
  corrected classification, corrected timing, note, or created-event reference
- provenance:
  which surface produced the review, from which workflow, and when
- training eligibility:
  whether the decision is allowed to flow into positive or negative dataset
  generation

At the decision level, the system must represent:
- `verified`
- `rejected`
- `relabeled`
- `boundary_corrected`
- `missed_event_added`

Compatibility note:
- the current Foundry `review_outcome` and nested `ReviewDecision` shape should
  be preserved during migration where practical
- richer decision kinds may be added before any compatibility field is removed

## Positive And Negative Signal Rules

These rules are the core product safety bar.

Positive signal is created only from:
- `verified`
- `missed_event_added`
- the corrected side of `relabeled`
- the corrected side of `boundary_corrected` when the corrected clip/window is
  explicitly materialized

Negative signal is created only from:
- `rejected`
- the original predicted side of `relabeled`
- the original predicted side of `boundary_corrected` when the original window
  is explicitly materialized

No signal is created from:
- untouched events
- pending queue items
- partially edited events that were never explicitly committed as review
- bulk inference output by itself

## Surface Behavior

### Phone Review

The phone review page is the lightweight explicit-review surface.

V1 phone behavior:
- one page
- one current review target at a time
- project/song/layer scoped queues
- explicit `Prev` and `Next`
- replay current clip
- verify, reject, and relabel
- clear progress/counter state

V1 phone behavior does not need to include:
- freeform timeline-style retiming
- manual event creation on device
- alternate truth mutation outside the shared review contract

### Timeline Fix Mode

Timeline fix mode is the dense expert correction surface.

Timeline actions that must emit review signal:
- explicit verify or approve-as-correct
- delete false positive
- relabel or move event between classified layers
- resize or retime an event
- manually add a missing event

Critical constraint:
- timeline review emission must pass through the canonical path
  `run_echozero.py` -> `echozero/ui/qt/app_shell.py` ->
  `echozero/application/timeline/*`
- no widget-only review persistence

## Queue Model

Review queues are not the same thing as review signal records.

Queue responsibilities:
- define the ordered worklist
- define filter context such as song, layer, class, polarity, and outcome
- provide deterministic next/previous navigation
- provide enough audio/context to let the operator decide quickly

Signal responsibilities:
- preserve the durable reviewed truth
- outlive any one phone session or queue generation run
- feed downstream dataset export and training filters

This split matters because a queue can be rebuilt many times while the review
signal record remains the durable artifact.

## Provenance Requirements

Every durable review signal must retain enough provenance to answer:
- where did this review target come from
- what did the model predict at review time
- which surface produced the correction
- what operator action created the record
- what main event or manually created event is now the canonical result

Minimum provenance fields should include:
- `surface`
- `project_ref`
- `song_ref`
- `layer_ref`
- `event_ref`
- `source_event_ref` when a derived event points back to an original event
- `model_ref`
- `reviewed_at`

## Training Consumption

Foundry is the downstream consumer of review signal.

The initial downstream jobs are:
- export reviewed positives and negatives into dataset-ready form
- preserve relabel pairs and corrected-boundary provenance
- measure reviewed volume and class balance before training
- support batch promotion into Foundry datasets or run inputs

The review feature is complete before project-adaptive models only when:
- reviewed signal can be created reliably
- reviewed signal can be promoted reliably
- reviewed signal can be audited after the fact

## Canonical Ownership

- Timeline truth mutation and operator actions:
  `echozero/application/timeline/*`
- App-shell integration and launch surfaces:
  `echozero/ui/qt/app_shell.py` and related Qt mixins
- Persisted review queue and review signal storage:
  `echozero/foundry/*`
- Training export and downstream run preparation:
  `echozero/foundry/services/*`

## Acceptance Criteria

This feature is done when all of the following are true:

- the phone app and timeline fix mode write the same canonical review decision
  vocabulary
- only explicit verified positives are promoted as positive training signal
- rejected, relabeled, boundary-corrected, and missed-event decisions preserve
  enough provenance for later audit
- a project/song/layer review queue can be generated from canonical EZ data
- timeline fix mode emits durable review records through the application
  boundary, not widget-local state
- Foundry can consume reviewed records into dataset-ready exports without
  guessing at operator intent

## Relationship To Older Decisions

This spec is consistent with the intent of D129, D146, and D149 in
[architecture/DECISIONS.md](architecture/DECISIONS.md), but it tightens the
operational rule:

- historical direction:
  user correction should feed model improvement
- tightened rule here:
  only explicit reviewed positives become positive signal

That tightening is deliberate.
It prevents the system from learning from silence, drift, or accidental
inaction.
