"""Direct seam coverage for object-action settings mixins.
Exists to freeze the extracted context and persistence helpers behind the service root.
Connects cleanup-oriented mixin splits to small application-level regression tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace

from echozero.application.timeline.object_action_settings_context_mixin import (
    ObjectActionSettingsContextMixin,
)
from echozero.application.timeline.object_action_settings_persistence_mixin import (
    ObjectActionSettingsPersistenceMixin,
)
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor
from echozero.persistence.entities import LayerRecord
from echozero.services.orchestrator import AnalysisResult


class _ContextShell(ObjectActionSettingsContextMixin):
    def __init__(self, active_run_lookup=None) -> None:
        self._active_run_lookup = active_run_lookup

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]:
        del action_id
        descriptor = ActionDescriptor(
            action_id="timeline.extract_stems",
            label="Extract Stems",
            object_types=("layer",),
            params_schema={"layer_id": "required"},
            workflow_id="layer.audio.extract_stems",
            pipeline_template_id="stem_separation",
        )
        return descriptor, "stem_separation"


class _DirtyTracker:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_dirty(self, record_id: str) -> None:
        self.marked.append(record_id)


class _LayerRepository:
    def __init__(self, records: dict[str, LayerRecord]) -> None:
        self.records = records

    def get(self, layer_id: str) -> LayerRecord | None:
        return self.records.get(layer_id)

    def update(self, record: LayerRecord) -> None:
        self.records[record.id] = record


class _ProjectStorage:
    def __init__(self, records: dict[str, LayerRecord]) -> None:
        self.dirty_tracker = _DirtyTracker()
        self.layers = _LayerRepository(records)

    def transaction(self):
        return nullcontext()


class _Session:
    def __init__(
        self,
        *,
        active_song_id: str | None = None,
        active_song_version_id: str | None = None,
    ) -> None:
        self.active_song_id = active_song_id
        self.active_song_version_id = active_song_version_id


class _PersistenceShell(ObjectActionSettingsPersistenceMixin):
    def __init__(self, *, storage: _ProjectStorage, session: _Session) -> None:
        self._project_storage = storage
        self._session = session

    @property
    def project_storage(self) -> _ProjectStorage:
        return self._project_storage

    @property
    def session(self) -> _Session:
        return self._session

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]:
        del action_id
        descriptor = ActionDescriptor(
            action_id="timeline.extract_stems",
            label="Extract Stems",
            object_types=("layer",),
            params_schema={"layer_id": "required"},
            workflow_id="layer.audio.extract_stems",
            pipeline_template_id="stem_separation",
        )
        return descriptor, "stem_separation"


def _layer_record(
    *, layer_id: str, song_version_id: str, source_layer_id: str | None = None
) -> LayerRecord:
    provenance = {}
    if source_layer_id is not None:
        provenance["source_layer_id"] = source_layer_id
    return LayerRecord(
        id=layer_id,
        song_version_id=song_version_id,
        name=layer_id,
        layer_type="analysis",
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
        provenance=provenance,
    )


def test_object_action_context_resolve_params_normalizes_blank_layer_id() -> None:
    resolved = _ContextShell._resolve_params(
        "timeline.extract_stems",
        {"layer_id": "   "},
        object_id=" source_audio ",
        object_type="layer",
    )

    assert resolved["layer_id"] == "source_audio"


def test_object_action_context_rejects_missing_required_layer() -> None:
    shell = _ContextShell()

    try:
        shell._resolve_execution_context(
            "timeline.extract_stems",
            {},
            object_id=None,
            object_type="layer",
        )
    except ValueError as exc:
        assert str(exc) == "timeline.extract_stems requires a target layer."
    else:
        raise AssertionError("expected missing-layer validation to raise")


def test_object_action_context_uses_active_run_lookup_when_available() -> None:
    observed: list[tuple[str, object | None, str | None]] = []

    def _lookup(
        action_id: str, object_id: object | None, object_type: str | None
    ) -> dict[str, str]:
        observed.append((action_id, object_id, object_type))
        return {"status": "running"}

    shell = _ContextShell(active_run_lookup=_lookup)

    result = shell._lookup_active_run(
        "timeline.extract_stems",
        object_id="source_audio",
        object_type="layer",
    )

    assert result == {"status": "running"}
    assert observed == [("timeline.extract_stems", "source_audio", "layer")]


def test_object_action_persistence_marks_song_default_scope_dirty_by_song() -> None:
    storage = _ProjectStorage({})
    shell = _PersistenceShell(
        storage=storage,
        session=_Session(active_song_id="song_alpha", active_song_version_id="version_alpha"),
    )
    config = SimpleNamespace(
        song_id="song_alpha",
        song_version_id="version_alpha",
    )

    shell._mark_scope_persist_dirty(scope="song_default", config=config)

    assert storage.dirty_tracker.marked == ["song_alpha"]


def test_object_action_persistence_updates_generated_layer_provenance_and_marks_dirty() -> None:
    generated = _layer_record(layer_id="generated_layer", song_version_id="version_alpha")
    untouched = _layer_record(
        layer_id="untouched_layer",
        song_version_id="version_beta",
        source_layer_id="source_audio",
    )
    storage = _ProjectStorage(
        {
            generated.id: generated,
            untouched.id: untouched,
        }
    )
    shell = _PersistenceShell(
        storage=storage,
        session=_Session(active_song_id="song_alpha", active_song_version_id="version_alpha"),
    )

    shell.persist_generated_source_layer_id(
        analysis_result=AnalysisResult(
            song_version_id="version_alpha",
            pipeline_id="stem_separation",
            layer_ids=["generated_layer", "untouched_layer", "missing_layer"],
            take_ids=[],
            duration_ms=10.0,
        ),
        source_layer_id="source_audio",
    )

    assert storage.layers.get("generated_layer") is not None
    assert storage.layers.get("generated_layer").provenance["source_layer_id"] == "source_audio"
    assert storage.layers.get("untouched_layer").provenance["source_layer_id"] == "source_audio"
    assert storage.dirty_tracker.marked == ["version_alpha"]


def test_object_action_persistence_requires_active_song_and_version() -> None:
    shell = _PersistenceShell(storage=_ProjectStorage({}), session=_Session())

    try:
        shell._require_active_song_id("save settings")
    except RuntimeError as exc:
        assert str(exc) == "save settings requires an active song."
    else:
        raise AssertionError("expected missing active song to raise")

    try:
        shell._require_active_song_version_id("save settings")
    except RuntimeError as exc:
        assert str(exc) == "save settings requires an active song version."
    else:
        raise AssertionError("expected missing active song version to raise")
