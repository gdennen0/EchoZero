"""Phase 3 timeline performance guardrail benchmark.

Usage:
  python tests/benchmarks/benchmark_timeline_phase3.py [--strict] [--json-out path]

This benchmark focuses on two high-risk hot paths:
1) Cached timeline assembly throughput (transport-only updates)
2) Event-lane paint cost with dense event timelines
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import EventPresentation
from echozero.application.session.models import Session
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation


DEFAULT_THRESHOLDS_MS = {
    "assemble_cached_p95_ms": 2.5,
    "assemble_cached_max_ms": 6.0,
    "event_lane_p95_ms": 8.0,
    "event_lane_max_ms": 16.0,
}

DEFAULT_THRESHOLDS_PATH = Path("tests") / "benchmarks" / "timeline_phase3_thresholds.json"


def _event(event_id: str, take_id: str, t: float) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=t,
        end=t + 0.08,
        label="hit",
        color="#66a3ff",
    )


def _build_dense_timeline(*, layer_count: int = 10, events_per_main: int = 1200) -> Timeline:
    layers: list[Layer] = []
    spacing = 0.09
    for layer_idx in range(layer_count):
        main_take_id = f"take_main_{layer_idx}"
        alt_take_id = f"take_alt_{layer_idx}"

        main_events = [_event(f"m_{layer_idx}_{i}", main_take_id, i * spacing) for i in range(events_per_main)]
        alt_events = [_event(f"a_{layer_idx}_{i}", alt_take_id, (i * spacing) + 0.02) for i in range(max(100, events_per_main // 4))]

        layer = Layer(
            id=LayerId(f"layer_{layer_idx}"),
            timeline_id=TimelineId("timeline_perf"),
            name=f"Layer {layer_idx}",
            kind=LayerKind.EVENT,
            order_index=layer_idx,
            takes=[
                Take(id=TakeId(main_take_id), layer_id=LayerId(f"layer_{layer_idx}"), name="Main", events=main_events),
                Take(id=TakeId(alt_take_id), layer_id=LayerId(f"layer_{layer_idx}"), name="Take 2", events=alt_events),
            ],
        )
        layers.append(layer)

    timeline = Timeline(
        id=TimelineId("timeline_perf"),
        song_version_id=SongVersionId("version_perf"),
        layers=layers,
    )
    timeline.selection.selected_layer_id = layers[0].id
    timeline.selection.selected_take_id = layers[0].takes[1].id
    timeline.viewport.pixels_per_second = 180.0
    timeline.viewport.scroll_x = 0.0
    return timeline


def _build_session() -> Session:
    return Session(
        id=SessionId("session_perf"),
        project_id=ProjectId("project_perf"),
        active_song_id=SongId("song_perf"),
        active_song_version_id=SongVersionId("version_perf"),
        active_timeline_id=TimelineId("timeline_perf"),
    )


def _ms(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_idx = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "count": float(len(samples)),
        "avg_ms": float(sum(samples) / len(samples)),
        "p50_ms": float(statistics.median(samples)),
        "p95_ms": float(ordered[p95_idx]),
        "max_ms": float(max(samples)),
    }


def benchmark_assemble_cached(*, iterations: int = 180) -> dict[str, float]:
    timeline = _build_dense_timeline()
    session = _build_session()
    assembler = TimelineAssembler()

    # warm
    assembler.assemble(timeline, session)

    samples: list[float] = []
    for i in range(iterations):
        session.transport_state.playhead = float(i) * 0.05  # transport-only mutation
        t0 = time.perf_counter()
        assembler.assemble(timeline, session)
        samples.append((time.perf_counter() - t0) * 1000.0)

    return _ms(samples)


def benchmark_event_lane_paint(*, iterations: int = 220) -> dict[str, float]:
    events: list[EventPresentation] = []
    for i in range(20000):
        start = i * 0.015
        events.append(
            EventPresentation(
                event_id=EventId(f"e{i}"),
                start=start,
                end=start + 0.08,
                label="evt",
                color="#66a3ff",
                is_selected=False,
            )
        )

    presentation = EventLanePresentation(
        layer_id=LayerId("layer_render"),
        events=events,
        pixels_per_second=180.0,
        scroll_x=9000.0,
        header_width=320,
        event_height=22,
        dimmed=False,
        viewport_width=1500,
    )

    app = QApplication.instance() or QApplication([])
    image = QImage(1500, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    block = EventLaneBlock()

    samples: list[float] = []
    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            block.paint(painter, 10, presentation)
            samples.append((time.perf_counter() - t0) * 1000.0)
    finally:
        painter.end()
        app.processEvents()

    return _ms(samples)


def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        path = DEFAULT_THRESHOLDS_PATH
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS_MS)

    payload = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_THRESHOLDS_MS)
    for key in out.keys():
        if key in payload:
            out[key] = float(payload[key])
    return out


def run(strict: bool = False, thresholds_path: Path | None = None) -> dict:
    thresholds = _load_thresholds(thresholds_path)

    assemble = benchmark_assemble_cached()
    event_lane = benchmark_event_lane_paint()

    checks = {
        "assemble_cached_p95_ms": assemble["p95_ms"],
        "assemble_cached_max_ms": assemble["max_ms"],
        "event_lane_p95_ms": event_lane["p95_ms"],
        "event_lane_max_ms": event_lane["max_ms"],
    }

    failures: list[str] = []
    for key, value in checks.items():
        if value > thresholds[key]:
            failures.append(f"{key}: {value:.3f}ms > {thresholds[key]:.3f}ms")

    payload = {
        "thresholds_ms": thresholds,
        "assemble_cached": assemble,
        "event_lane_paint": event_lane,
        "checks_ms": checks,
        "pass": not failures,
        "failures": failures,
    }

    print(json.dumps(payload, indent=2))

    if strict and failures:
        raise SystemExit(2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 timeline performance guardrails")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when thresholds are exceeded")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write benchmark JSON")
    parser.add_argument("--thresholds", type=str, default="", help="Optional thresholds JSON path")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    payload = run(strict=args.strict, thresholds_path=Path(args.thresholds) if args.thresholds else None)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
