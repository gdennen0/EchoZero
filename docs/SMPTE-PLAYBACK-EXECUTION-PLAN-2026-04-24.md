# SMPTE Playback Execution Plan

Last updated: 2026-04-24

Primary sources:

1. [SMPTE-PLAYBACK-RFC-SKELETON-2026-04-24.md](/Users/march/Documents/GitHub/EchoZero/docs/SMPTE-PLAYBACK-RFC-SKELETON-2026-04-24.md:1)
2. [SMPTE-PLAYBACK-TASK-BOARD-2026-04-24.md](/Users/march/Documents/GitHub/EchoZero/docs/SMPTE-PLAYBACK-TASK-BOARD-2026-04-24.md:1)

## Strategy

Do not execute SMPTE work as one giant shot. Execute in gated phases so timing-contract errors are trapped before they spread into UI/export/sync and MA3-facing paths.

## Immediate Order

1. Contract freeze tasks (`SP-00` through `SP-05`)
2. Shared codec and contract convergence (`SP-10` through `SP-15`)
3. Master mode generation (`SP-20` through `SP-24`)
4. Chase mode (`SP-30` through `SP-34`)
5. Hardening and signoff (`SP-40` through `SP-50`)

## Why This Order

1. SMPTE reliability depends on a stable timebase contract first.
2. UI/export/sync convergence must happen before protocol IO so every path uses one conversion policy.
3. Master mode is simpler than chase and gives us protocol and drift visibility before control-loop complexity.
4. Chase mode should only land after codec and observability are proven.

## Active Slice (Start Now)

Execute this bounded slice first:

1. SP-00: freeze supported v1 matrix
2. SP-01: freeze MTC/LTC scope by phase
3. SP-02: freeze SLO baseline
4. SP-03: freeze chase lock-state contract
5. SP-04: freeze `TimebaseSpec` typed contract
6. SP-05: complete requirement-to-proof matrix

Definition of done for active slice:

1. no unresolved contract ambiguity for codec implementation
2. no placeholder SLO text in active baseline sections
3. every requirement is mapped to proof

## Next Slice (After Active Slice)

1. implement canonical `TimecodeCodec`
2. migrate UI labels/ruler to codec
3. migrate export conversion to codec
4. migrate sync-facing conversion semantics to codec
5. run phase-1 gate suite

## Risks To Watch Early

1. hidden duplicate conversion helpers outside obvious playback/export paths
2. boundary semantics mismatches (`seek`/`stop` vs frame labels)
3. legacy assumptions around `timecode_fps` persistence values

## Reporting Cadence

For each phase gate:

1. list task IDs completed
2. list proof lanes run
3. list residual risk/open defects
4. explicitly label human-path vs synthetic proof

