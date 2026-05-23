# Timeline Event 80 ms Investigation Prompt

Status: draft
Last verified: 2026-05-20
Lane: Foundry

## Prompt

You are working in the local EchoZero repository on this machine.

Active objective: investigate why timeline Events and exported review samples are all or mostly 80 ms long, and identify the smallest correct fix path.

Active lane: Foundry.
Excluded lanes: MA3 harness, unrelated EZ app UI work, broad architecture cleanup.

Follow `AGENTS.md` and local repo instructions. Open `docs/STATUS.md` if you need current repo truth, and open subsystem docs only when the task needs them.

## Context

The Noah Kahan kick/snare model experiment exposed a likely training-data shape problem:

- Shared review samples under `${ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT:-~/.echozero/data/tmp/review_samples}/<class>/` are mostly 80 ms audio clips.
- The manifest entries show `end_seconds - start_seconds == 0.08` for many exported clips.
- `echozero/ui/qt/timeline_review_sample_export.py` appears to pass Event `start_seconds` and `end_seconds` through to `ReviewAudioClipService.materialize_event_clip`.
- `ReviewAudioClipService.materialize_event_clip` appears to write the requested time range rather than imposing 80 ms itself.
- Existing Olivia Monster model-building code used runtime-shaped clips around events, not only the tiny Event span.
- Training on 80 ms clips padded to model input length likely mismatches runtime inference, which evaluates a longer window from the song or stem.

Working hypothesis: the exporter is faithfully preserving timeline Event durations, and the real question is why drum Events are being created, normalized, corrected, or presented as 80 ms regions.

## What To Investigate

Trace the source of Event duration for drum classifications from processor output through timeline persistence, app review, correction, and review-sample export.

Start with targeted searches around:

- `0.08`
- `80`
- `duration`
- `end_seconds`
- `Event`
- `classification`
- `ReviewAudioClipService`
- `timeline_review_sample_export`
- `binary_drum_classify`
- `LABEL_MIN_SEPARATION_MS`

Known candidates from an initial scan:

- `echozero/ui/qt/timeline/real_data_fixture.py` contains `hit.time + 0.08` and `max(event.duration, 0.08)`, but this may be fixture/support only.
- `echozero/processors/binary_drum_classify.py` contains `LABEL_MIN_SEPARATION_MS["kick"] = 80.0`, which sounds like separation/debounce, not Event length.
- `echozero/ui/qt/timeline/app_shell_timeline_review.py` passes Event ranges or corrected ranges into sample export.

Please determine:

1. Where drum Event `end` / duration is first assigned.
2. Whether 80 ms is a true semantic Event length, a display minimum, a processor default, a review correction default, or an export artifact.
3. Whether “sample length” should mean the actual source/audio event span, a runtime training window centered around the Event, or both as separate concepts.
4. Which code path should change so future review sample exports preserve model-useful audio while timeline Events remain visually and semantically correct.
5. Whether existing exported 80 ms shared review samples should be regenerated from source audio and manifest timing rather than reused.

## Guardrails

- Do not change MA3 code.
- Do not redesign unrelated timeline UI.
- Do not re-train models as part of this investigation.
- Do not commit generated audio, model artifacts, caches, or local state.
- If you make code changes, keep them narrowly scoped and add focused tests.
- If the correct fix is larger than one small patch, document the recommended implementation path instead of half-building it.

## Evidence To Collect

Please include file/line references for:

- Event creation or mapping code that assigns duration.
- Any fixed 80 ms constants and whether they are used for duration, separation, display, selection, or export.
- Export path from reviewed/corrected Event to WAV clip.
- Any tests that lock in 80 ms behavior.

If local sample files are present, measure a small deterministic subset and report:

- class folder counts
- min/p50/p90/max duration by class
- whether manifest timing matches file duration

## Desired Output

Return a final payload with:

```text
status: success | blocked | partial
changed_files:
  - path
tests_run:
  - command/result
summary:
  - concise bullets
root_cause:
  - concise bullets with file/line references
recommended_fix:
  - concise bullets
blocker: null or explanation
residual_risk:
  - concise bullets
```

If you make no code changes, use `changed_files: []`.

