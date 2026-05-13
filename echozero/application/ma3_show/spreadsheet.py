"""Spreadsheet import helpers for clean-sheet MA3 show planning.
Exists to turn operator setlist sheets into EchoZero show intent.
Connects CSV metadata, section rows, and setlist song models.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from echozero.application.ma3_show.models import Setlist, ShowSong, SongSection


_COLUMN_ALIASES = {
    "title": {"title", "song", "song title", "name"},
    "artist": {"artist", "act"},
    "order": {"order", "position", "setlist order", "set order", "#"},
    "notes": {"notes", "note", "setlist notes", "song notes"},
    "bpm": {"bpm", "tempo"},
    "section": {"section", "song section", "part"},
    "section_time": {"section time", "time", "start", "start time", "start seconds"},
    "cue": {"cue", "cue number", "cue ref", "cue_ref"},
    "main_sequence_no": {"main sequence", "main sequence no", "sequence", "sequence block"},
    "page": {"page", "ma3 page"},
    "executor": {"executor", "exec", "ma3 executor"},
    "timecode_pool": {"timecode", "timecode pool", "tc", "tc pool"},
}


@dataclass(frozen=True, slots=True)
class SpreadsheetSongRow:
    """Normalized spreadsheet row used as show intent."""

    row_number: int
    title: str
    artist: str = ""
    order: int | None = None
    notes: str = ""
    bpm: float | None = None
    section_label: str | None = None
    section_start_seconds: float | None = None
    cue_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpreadsheetImportResult:
    """Result of importing setlist intent from a spreadsheet."""

    setlist: Setlist
    rows: tuple[SpreadsheetSongRow, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def import_setlist_csv(path: str | Path, *, setlist_id: str = "imported") -> SpreadsheetImportResult:
    """Import a CSV setlist as EchoZero show intent."""

    source_path = Path(path)
    rows: list[SpreadsheetSongRow] = []
    warnings: list[str] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV setlist requires a header row")
        field_map = _field_map(reader.fieldnames)
        for row_number, raw_row in enumerate(reader, start=2):
            normalized = _normalize_row(raw_row, field_map)
            title = str(normalized.get("title") or "").strip()
            if not title:
                warnings.append(f"Row {row_number} skipped: missing title")
                continue
            rows.append(
                SpreadsheetSongRow(
                    row_number=row_number,
                    title=title,
                    artist=str(normalized.get("artist") or "").strip(),
                    order=_optional_int(normalized.get("order")),
                    notes=str(normalized.get("notes") or "").strip(),
                    bpm=_optional_float(normalized.get("bpm")),
                    section_label=_optional_text(normalized.get("section")),
                    section_start_seconds=_optional_float(normalized.get("section_time")),
                    cue_ref=_optional_text(normalized.get("cue")),
                    metadata=dict(raw_row),
                )
            )

    songs_by_key: dict[tuple[str, str, int], ShowSong] = {}
    row_order = 0
    for row in rows:
        order = row.order if row.order is not None else row_order
        key = (_slug(row.title), _slug(row.artist), int(order))
        existing = songs_by_key.get(key)
        section = _section_from_row(row, existing_section_count=len(existing.sections) if existing else 0)
        if existing is None:
            songs_by_key[key] = ShowSong(
                id=_song_id(row.title, row.artist, order),
                title=row.title,
                artist=row.artist,
                order=order,
                notes=row.notes,
                bpm=row.bpm,
                metadata={"spreadsheet_rows": [row.metadata]},
                sections=tuple([section] if section is not None else []),
            )
            row_order += 1
            continue
        sections = existing.sections + tuple([section] if section is not None else [])
        metadata = dict(existing.metadata)
        metadata["spreadsheet_rows"] = list(metadata.get("spreadsheet_rows") or []) + [
            row.metadata
        ]
        songs_by_key[key] = ShowSong(
            id=existing.id,
            title=existing.title,
            artist=existing.artist,
            order=existing.order,
            notes=existing.notes or row.notes,
            bpm=existing.bpm if existing.bpm is not None else row.bpm,
            metadata=metadata,
            sections=sections,
            ma3_mapping=existing.ma3_mapping,
        )

    setlist = Setlist(
        id=setlist_id,
        name=source_path.stem,
        songs=tuple(songs_by_key.values()),
        metadata={"source_path": str(source_path)},
    )
    return SpreadsheetImportResult(setlist=setlist, rows=tuple(rows), warnings=tuple(warnings))


def _field_map(fieldnames: list[str]) -> dict[str, str]:
    field_map: dict[str, str] = {}
    for fieldname in fieldnames:
        normalized = _normalize_header(fieldname)
        for canonical, aliases in _COLUMN_ALIASES.items():
            if normalized in aliases:
                field_map[fieldname] = canonical
                break
        else:
            field_map[fieldname] = fieldname
    return field_map


def _normalize_row(raw_row: dict[str, str], field_map: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for source_key, value in raw_row.items():
        target_key = field_map.get(source_key, source_key)
        normalized[target_key] = value
    return normalized


def _section_from_row(row: SpreadsheetSongRow, *, existing_section_count: int) -> SongSection | None:
    if row.section_label is None and row.section_start_seconds is None and row.cue_ref is None:
        return None
    section_no = existing_section_count + 1
    return SongSection(
        id=f"section-{section_no}",
        label=row.section_label or f"Cue {row.cue_ref or section_no}",
        start_seconds=row.section_start_seconds or 0.0,
        cue_number=row.cue_ref,
        cue_ref=row.cue_ref,
        notes=row.notes,
    )


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " "))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")


def _song_id(title: str, artist: str, order: int) -> str:
    base = "-".join(part for part in (_slug(title), _slug(artist)) if part)
    return f"song-{order + 1}-{base or 'untitled'}"


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
