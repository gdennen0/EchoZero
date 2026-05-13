"""Visual Lab catalog model.
Exists to make every previewable UI object addressable by stable id and folder.
The model is support-only; production EchoZero UI may migrate components into it over time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from PyQt6.QtWidgets import QWidget

from dev.visual_lab.tokens import TokenFieldSpec, VisualLabTokens, token_field_specs

RenderFactory = Callable[[VisualLabTokens], QWidget]
SourceKind = Literal["production-backed", "current-model synthetic", "lab-only experimental"]


@dataclass(frozen=True, slots=True)
class StylePropertySpec:
    """One editable property on a style-addressable component part."""

    name: str
    token_path: str


@dataclass(frozen=True, slots=True)
class StyleTargetSpec:
    """Nested component part with editable style properties."""

    component: str
    part_path: str
    label: str
    properties: tuple[StylePropertySpec, ...]

    @property
    def target_id(self) -> str:
        return f"{self.component}.{self.part_path}"

    @property
    def token_paths(self) -> tuple[str, ...]:
        return tuple(property_spec.token_path for property_spec in self.properties)


@dataclass(frozen=True, slots=True)
class ResolvedStyleTarget:
    """Style target with token field metadata resolved for the active token state."""

    component: str
    part_path: str
    label: str
    specs: tuple[TokenFieldSpec, ...]

    @property
    def target_id(self) -> str:
        return f"{self.component}.{self.part_path}"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One independently previewable Visual Lab object."""

    entry_id: str
    name: str
    category: str
    description: str
    kind: str
    source_kind: SourceKind
    source_path: str
    render: RenderFactory
    part_ids: tuple[str, ...] = ()
    editable_token_paths: tuple[str, ...] = ()
    style_targets: tuple[StyleTargetSpec, ...] = ()


class CatalogRegistry:
    """Ordered folder-like registry for Visual Lab entries."""

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        ordered_entries = tuple(entries)
        by_id: dict[str, CatalogEntry] = {}
        for entry in ordered_entries:
            if entry.entry_id in by_id:
                raise ValueError(f"duplicate Visual Lab catalog id: {entry.entry_id}")
            by_id[entry.entry_id] = entry
        if not ordered_entries:
            raise ValueError("Visual Lab catalog requires at least one entry")
        self._entries = ordered_entries
        self._by_id = by_id

    @property
    def entries(self) -> tuple[CatalogEntry, ...]:
        return self._entries

    def categories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.category for entry in self._entries))

    def entries_for_category(self, category: str) -> tuple[CatalogEntry, ...]:
        return tuple(entry for entry in self._entries if entry.category == category)

    def get(self, entry_id: str) -> CatalogEntry:
        try:
            return self._by_id[entry_id]
        except KeyError as exc:
            raise KeyError(f"unknown Visual Lab catalog id: {entry_id}") from exc

    def first(self) -> CatalogEntry:
        return self._entries[0]

    def previous_id(self, entry_id: str) -> str:
        index = self._index(entry_id)
        return self._entries[(index - 1) % len(self._entries)].entry_id

    def next_id(self, entry_id: str) -> str:
        index = self._index(entry_id)
        return self._entries[(index + 1) % len(self._entries)].entry_id

    def validate_parts(self) -> None:
        for entry in self._entries:
            for part_id in entry.part_ids:
                if part_id not in self._by_id:
                    raise ValueError(f"{entry.entry_id} references unknown catalog part {part_id}")

    def validate_editable_tokens(self, tokens: VisualLabTokens) -> None:
        """Ensure every entry editable token points at a known lab token."""
        known_paths = {spec.path for spec in token_field_specs(tokens)}
        for entry in self._entries:
            for path in _entry_token_paths(entry):
                if path not in known_paths:
                    raise ValueError(
                        f"{entry.entry_id} references unknown editable token {path}"
                    )

    def editable_specs_for(
        self, entry_id: str, tokens: VisualLabTokens
    ) -> tuple[TokenFieldSpec, ...]:
        """Return token editor fields that are relevant to one catalog entry."""
        entry = self.get(entry_id)
        specs_by_path = {spec.path: spec for spec in token_field_specs(tokens)}
        paths = tuple(dict.fromkeys(_entry_token_paths(entry)))
        return tuple(specs_by_path[path] for path in paths)

    def editable_targets_for(
        self, entry_id: str, tokens: VisualLabTokens
    ) -> tuple[ResolvedStyleTarget, ...]:
        """Return nested style targets for a selected catalog entry."""
        entry = self.get(entry_id)
        specs_by_path = {spec.path: spec for spec in token_field_specs(tokens)}
        targets = entry.style_targets or (
            StyleTargetSpec(
                component=entry.entry_id,
                part_path="root",
                label=entry.name,
                properties=tuple(
                    StylePropertySpec(path.rsplit(".", 1)[-1], path)
                    for path in entry.editable_token_paths
                ),
            ),
        )
        return tuple(
            ResolvedStyleTarget(
                component=target.component,
                part_path=target.part_path,
                label=target.label,
                specs=tuple(
                    specs_by_path[path]
                    for path in dict.fromkeys(target.token_paths)
                    if path in specs_by_path
                ),
            )
            for target in targets
        )

    def _index(self, entry_id: str) -> int:
        self.get(entry_id)
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return index
        raise AssertionError("catalog id lookup and ordered entries diverged")


def flatten_categories(
    entries_by_category: Sequence[tuple[str, Sequence[CatalogEntry]]],
) -> list[CatalogEntry]:
    """Return entries in folder order while ensuring each entry names its folder."""
    entries: list[CatalogEntry] = []
    for category, category_entries in entries_by_category:
        for entry in category_entries:
            if entry.category != category:
                raise ValueError(
                    f"{entry.entry_id} category {entry.category!r} does not match folder {category!r}"
                )
            entries.append(entry)
    return entries


def _entry_token_paths(entry: CatalogEntry) -> tuple[str, ...]:
    target_paths = tuple(
        token_path for target in entry.style_targets for token_path in target.token_paths
    )
    return entry.editable_token_paths + target_paths
