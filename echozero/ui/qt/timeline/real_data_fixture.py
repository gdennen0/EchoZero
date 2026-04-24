"""Build a timeline presentation from real audio analysis runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import echozero.pipelines.templates  # noqa: F401 - registers templates via decorators
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import FollowMode, LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.domain.types import AudioData, Event as DomainEvent, EventData
from echozero.ui.qt.timeline.drum_classifier_preview import classify_drum_hits
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, fixture_color, fixture_take_action_label
from echozero.ui.qt.timeline.waveform_cache import register_waveform_from_audio_file


@dataclass(slots=True)
class RealDataRunSummary:
    audio_path: Path
    working_dir: Path
    song_version_id: str
    layer_count: int
    take_count: int
    event_count_main: int


def build_real_data_presentation(
    audio_path: str | Path,
    *,
    working_root: str | Path,
    song_title: str = "Doechii Nissan Altima",
) -> tuple[TimelinePresentation, RealDataRunSummary]:
    from echozero.main import create_project

    source = Path(audio_path)
    root = Path(working_root)
    root.mkdir(parents=True, exist_ok=True)

    project = create_project("Timeline Real Data", working_dir_root=root)
    try:
        _, version = project.import_song(
            title=song_title,
            audio_source=source,
            artist="Doechii",
            default_templates=[],
        )

        register_waveform_from_audio_file("song-real", source)

        # Stems-first flow: separate track into drums/bass/vocals/other before classifier work.
        project.analyze(
            version.id,
            "stem_separation",
            {"model": "htdemucs", "device": "cpu", "shifts": 0},
        )

        layers = project.storage.layers.list_by_version(version.id)
        presentation_layers: list[LayerPresentation] = [
            LayerPresentation(
                layer_id=LayerId("source_audio"),
                title="Song",
                subtitle=f"Real import · {source.name}",
                kind=LayerKind.AUDIO,
                is_selected=False,
                is_expanded=False,
                color=fixture_color("song"),
                waveform_key="song-real",
                badges=["main", "audio", "real-data"],
                status=LayerStatusPresentation(
                    stale=False,
                    manually_modified=False,
                    source_label="Imported track",
                    sync_label=TIMELINE_STYLE.fixture.default_sync_label,
                ),
            )
        ]

        total_takes = 0
        total_main_events = 0
        drums_audio_path: Path | None = None

        for order, layer_record in enumerate(layers, start=1):
            takes = project.storage.takes.list_by_layer(layer_record.id)
            if not takes:
                continue
            total_takes += len(takes)

            main_take = next((take for take in takes if take.is_main), takes[0])
            main_kind = LayerKind.EVENT if isinstance(main_take.data, EventData) else LayerKind.AUDIO

            main_events = (
                _event_presentations_from_take(main_take)
                if isinstance(main_take.data, EventData)
                else []
            )
            total_main_events += len(main_events)

            take_rows: list[TakeLanePresentation] = []
            for take in takes:
                if take.is_main:
                    continue
                take_kind = LayerKind.EVENT if isinstance(take.data, EventData) else LayerKind.AUDIO
                take_waveform_key = None
                if take_kind == LayerKind.AUDIO and isinstance(take.data, AudioData):
                    take_waveform_key = f"real-{layer_record.name}-{take.id}"
                    register_waveform_from_audio_file(take_waveform_key, take.data.file_path)

                take_rows.append(
                    TakeLanePresentation(
                        take_id=TakeId(str(take.id)),
                        name=take.label,
                        is_main=False,
                        kind=take_kind,
                        events=_event_presentations_from_take(take) if isinstance(take.data, EventData) else [],
                        source_ref=_source_ref(take.source),
                        waveform_key=take_waveform_key,
                        source_audio_path=str(take.data.file_path) if isinstance(take.data, AudioData) else None,
                        actions=[
                            TakeActionPresentation(
                                action_id="overwrite_main",
                                label=fixture_take_action_label("overwrite_main"),
                            ),
                            TakeActionPresentation(
                                action_id="merge_main",
                                label=fixture_take_action_label("merge_main"),
                            ),
                            TakeActionPresentation(
                                action_id="delete_take",
                                label=fixture_take_action_label("delete_take"),
                            ),
                        ],
                    )
                )

            status = LayerStatusPresentation(
                stale=bool(layer_record.state_flags.get("stale", False)),
                manually_modified=bool(layer_record.state_flags.get("manually_modified", False)),
                source_label=_source_label(layer_record),
                sync_label=TIMELINE_STYLE.fixture.default_sync_label,
            )
            main_waveform_key = None
            if main_kind == LayerKind.AUDIO and isinstance(main_take.data, AudioData):
                main_waveform_key = f"real-{layer_record.name}-main"
                register_waveform_from_audio_file(main_waveform_key, main_take.data.file_path)
                if layer_record.name.lower() == "drums":
                    drums_audio_path = Path(main_take.data.file_path)

            presentation_layers.append(
                LayerPresentation(
                    layer_id=LayerId(str(layer_record.id)),
                    title=layer_record.name.title(),
                    subtitle=f"Stem output · layer {order}",
                    kind=main_kind,
                    is_selected=(order == 1),
                    is_expanded=bool(take_rows),
                    events=main_events,
                    takes=take_rows,
                    color=layer_record.color or _fixture_layer_color(layer_record.name),
                    badges=["main", "stem", main_kind.value, "real-data"],
                    waveform_key=main_waveform_key,
                    source_audio_path=str(main_take.data.file_path) if isinstance(main_take.data, AudioData) else None,
                    status=status,
                )
            )

        # Preview classifier lanes from drums stem (stems-first progression).
        if drums_audio_path is not None and drums_audio_path.exists():
            preview_hits = classify_drum_hits(drums_audio_path, onset_threshold=0.025, min_gap=0.04)
            preview_layers = _classifier_layers_from_hits(preview_hits)
            presentation_layers.extend(preview_layers)
            total_main_events += sum(len(layer.events) for layer in preview_layers)

        timeline = TimelinePresentation(
            timeline_id=TimelineId("timeline_real_data"),
            title="Stage Zero Timeline · Real Data",
            layers=presentation_layers,
            playhead=8.0,
            is_playing=False,
            follow_mode=FollowMode.CENTER,
            selected_layer_id=presentation_layers[1].layer_id if len(presentation_layers) > 1 else presentation_layers[0].layer_id,
            selected_take_id=None,
            selected_event_ids=[],
            pixels_per_second=180.0,
            scroll_x=0.0,
            scroll_y=0.0,
            current_time_label=_fmt_time(8.0),
            end_time_label=_fmt_time(version.duration_seconds),
        )

        summary = RealDataRunSummary(
            audio_path=source,
            working_dir=project.storage.working_dir,
            song_version_id=version.id,
            layer_count=max(0, len(presentation_layers) - 1),
            take_count=total_takes,
            event_count_main=total_main_events,
        )
        return timeline, summary
    finally:
        project.close()


def build_real_data_variants(presentation: TimelinePresentation) -> dict[str, TimelinePresentation]:
    return {
        "real_default": presentation,
        "real_scrolled": replace(presentation, scroll_x=680.0, playhead=31.5, current_time_label=_fmt_time(31.5)),
        "real_zoomed_in": replace(presentation, pixels_per_second=300.0),
        "real_zoomed_out": replace(presentation, pixels_per_second=90.0),
    }


def _event_presentations_from_take(take) -> list[EventPresentation]:
    if not isinstance(take.data, EventData):
        return []

    events: list[DomainEvent] = []
    for layer in take.data.layers:
        events.extend(layer.events)

    events.sort(key=lambda event: (event.time, event.duration, str(event.id)))

    return [
        EventPresentation(
            event_id=EventId(str(event.id)),
            start=float(event.time),
            end=float(event.time + max(event.duration, 0.08)),
            label=_event_label(event),
        )
        for event in events
    ]


def _event_label(event: DomainEvent) -> str:
    if isinstance(event.classifications, dict) and event.classifications:
        first_key = next(iter(event.classifications.keys()))
        value = event.classifications.get(first_key)
        if isinstance(value, str) and value.strip():
            return value.strip().title()
        if isinstance(first_key, str) and first_key.strip():
            return first_key.strip().replace("_", " ").title()
    return "Onset"


def _source_label(layer_record) -> str:
    source = layer_record.source_pipeline or {}
    pipeline = source.get("pipeline_id", "pipeline")
    output_name = source.get("output_name", layer_record.name)
    return f"{pipeline} · {output_name}"


def _source_ref(source: Any) -> str | None:
    if source is None:
        return None
    run_id = getattr(source, "run_id", "")
    block_type = getattr(source, "block_type", "")
    if run_id and block_type:
        return f"{block_type}:{run_id[:8]}"
    if run_id:
        return str(run_id)
    return None


def _classifier_layers_from_hits(hits: dict[str, list]) -> list[LayerPresentation]:
    order = [
        ("kick", "Kick", fixture_color("kick")),
        ("snare", "Snare", fixture_color("snare")),
        ("hihat", "HiHat", fixture_color("hihat")),
        ("clap", "Clap", fixture_color("clap")),
    ]
    layers: list[LayerPresentation] = []

    for key, title, color in order:
        values = sorted(hits.get(key, []), key=lambda hit: hit.time)
        events = [
            EventPresentation(
                event_id=EventId(f"classifier_{key}_{i+1}"),
                start=float(hit.time),
                end=float(hit.time + 0.08),
                label=title,
                color=color,
            )
            for i, hit in enumerate(values)
        ]
        layers.append(
            LayerPresentation(
                layer_id=LayerId(f"classifier_{key}"),
                title=title,
                subtitle="Classifier preview from drums stem",
                kind=LayerKind.EVENT,
                is_selected=False,
                is_expanded=False,
                events=events,
                takes=[],
                color=color,
                badges=["main", "event", "classifier-preview", "real-data"],
                status=LayerStatusPresentation(
                    stale=False,
                    manually_modified=False,
                    source_label="Drums stem classifier preview",
                    sync_label=TIMELINE_STYLE.fixture.default_sync_label,
                ),
            )
        )

    return layers


def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


def _fixture_layer_color(name: str) -> str:
    token = name.strip().lower()
    return TIMELINE_STYLE.fixture.layer_color_tokens.get(token, TIMELINE_STYLE.fixture.fallback_audio_lane_hex)
