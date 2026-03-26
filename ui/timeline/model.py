"""
model.py — Pure Python data model for EchoZero 2 Timeline Prototype
NO Qt imports. Pure data structures only.
"""
from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class TimelineEvent:
    id: str
    time: float
    duration: float
    layer_id: str
    label: str
    color: tuple
    classifications: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimelineLayer:
    id: str
    name: str
    color: tuple
    order: int
    collapsed: bool = False
    height: float = 40.0


@dataclass
class TimelineState:
    events: List[TimelineEvent] = field(default_factory=list)
    layers: List[TimelineLayer] = field(default_factory=list)
    zoom_level: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    playhead_time: float = 0.0
    selection: Set[str] = field(default_factory=set)


@dataclass
class ViewportRect:
    time_start: float
    time_end: float
    layer_start: int
    layer_end: int


def visible_events(events, viewport_time_start, viewport_time_end):
    if not events:
        return []
    times = [e.time for e in events]
    right = bisect.bisect_right(times, viewport_time_end)
    return [e for e in events[:right] if e.time + e.duration > viewport_time_start]


_EVENT_NAMES = [
    "Attack", "Ambient", "Music", "Foley", "Dialog",
    "SFX", "Room Tone", "Score", "Effect", "Stinger",
    "Hit", "Drone", "Riser", "Transition",
]

_LAYER_COLORS = [
    (255, 90, 90), (90, 190, 255), (90, 255, 140), (255, 200, 70),
    (200, 90, 255), (255, 140, 70), (70, 220, 220), (255, 110, 200),
    (140, 255, 90), (170, 170, 255),
]


def generate_fake_data(num_events=500, num_layers=10, duration=300.0):
    rng = random.Random(42)
    layers = [
        TimelineLayer(id=f"layer_{i}", name=f"Layer {i+1}",
                      color=_LAYER_COLORS[i % len(_LAYER_COLORS)], order=i)
        for i in range(num_layers)
    ]
    layer_ids = [la.id for la in layers]
    layer_color_map = {la.id: la.color for la in layers}
    events = []
    for i in range(num_events):
        lid = rng.choice(layer_ids)
        br, bg, bb = layer_color_map[lid]
        start = rng.uniform(0.0, max(0.1, duration - 1.0))
        dur = rng.uniform(0.3, min(20.0, duration - start))
        color = (
            max(0, min(255, br + rng.randint(-25, 25))),
            max(0, min(255, bg + rng.randint(-25, 25))),
            max(0, min(255, bb + rng.randint(-25, 25))),
        )
        events.append(TimelineEvent(
            id=f"event_{i}", time=start, duration=dur, layer_id=lid,
            label=f"{rng.choice(_EVENT_NAMES)} {i}", color=color,
            classifications={"type": rng.choice(_EVENT_NAMES)},
        ))
    events.sort(key=lambda e: e.time)
    return TimelineState(events=events, layers=layers, zoom_level=100.0)
