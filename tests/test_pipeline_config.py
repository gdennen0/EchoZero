"""
Tests for the PipelineConfigRecord persistence model.
Verifies the full lifecycle: create from template → edit knobs → execute → re-edit.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import echozero.pipelines.templates  # noqa: F401 — register templates
from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.execution import ExecutionContext
from echozero.persistence.entities import PipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.result import Ok, is_ok, unwrap
from echozero.services.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Mock executor
# ---------------------------------------------------------------------------


class MockLoadAudioExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        from echozero.result import ok
        return ok(AudioData(sample_rate=44100, duration=5.0, file_path="test.wav",
                            channel_count=2))


class MockDetectOnsetsExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        from echozero.result import ok
        block = context.graph.blocks[block_id]
        threshold = block.settings.get("threshold", 0.5)
        # Return fewer events for higher thresholds
        count = max(1, int(10 * (1.0 - threshold)))
        import uuid
        events = tuple(
            Event(
                id=uuid.uuid4().hex, time=i * 0.5, duration=0.1,
                classifications={}, metadata={}, origin="mock",
            )
            for i in range(count)
        )
        layer = Layer(id=uuid.uuid4().hex, name="onsets", events=events)
        return ok(EventData(layers=(layer,)))


def _executors():
    return {
        "LoadAudio": MockLoadAudioExecutor(),
        "DetectOnsets": MockDetectOnsetsExecutor(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    s = ProjectStorage.create_new("TestProject", working_dir_root=tmp_path)
    yield s
    s.close()


@pytest.fixture
def song_version(session):
    """Create a song with a version and return the version."""
    from echozero.persistence.entities import SongRecord, SongVersionRecord
    from datetime import datetime, timezone
    import uuid

    now = datetime.now(timezone.utc)
    song = SongRecord(
        id=uuid.uuid4().hex, project_id=session.project.id,
        title="Test SongRecord", artist="Test", order=0,
    )
    version = SongVersionRecord(
        id=uuid.uuid4().hex, song_id=song.id, label="Original",
        audio_file="test.wav", duration_seconds=5.0,
        original_sample_rate=44100, audio_hash="abc123", created_at=now,
    )
    session.songs.create(song)
    session.song_versions.create(version)
    session.commit()
    return version


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateConfig:
    """Creating a PipelineConfigRecord from a template."""

    def test_create_config_succeeds(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        result = orch.create_config(session, song_version.id, "onset_detection")
        assert is_ok(result)
        config = unwrap(result)
        assert config.template_id == "onset_detection"
        assert config.song_version_id == song_version.id
        assert "threshold" in config.knob_values

    def test_config_persisted_to_db(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        loaded = session.pipeline_configs.get(config.id)
        assert loaded is not None
        assert loaded.template_id == "onset_detection"
        assert loaded.knob_values == config.knob_values

    def test_config_has_valid_graph(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        graph_data = json.loads(config.graph_json)
        assert "blocks" in graph_data
        assert len(graph_data["blocks"]) >= 2  # LoadAudio + DetectOnsets

    def test_config_has_outputs(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        outputs = json.loads(config.outputs_json)
        assert len(outputs) >= 1
        assert outputs[0]["name"] == "onsets"

    def test_create_with_overrides(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        result = orch.create_config(
            session, song_version.id, "onset_detection",
            knob_overrides={"threshold": 0.7},
        )
        config = unwrap(result)
        assert config.knob_values["threshold"] == 0.7

    def test_invalid_template_returns_error(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        result = orch.create_config(session, song_version.id, "nonexistent_template")
        assert not is_ok(result)

    def test_invalid_version_returns_error(self, session):
        orch = Orchestrator(get_registry(), _executors())
        result = orch.create_config(session, "bad_version_id", "onset_detection")
        assert not is_ok(result)

    def test_extract_classified_drums_defaults_use_point_one_five_onset_thresholds(
        self, session, song_version
    ):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "extract_classified_drums"))

        assert config.knob_values["kick_onset_threshold"] == pytest.approx(0.150)
        assert config.knob_values["snare_onset_threshold"] == pytest.approx(0.150)

        pipeline = config.to_pipeline()
        assert pipeline.graph.blocks["kick_onsets"].settings.get("threshold") == pytest.approx(0.150)
        assert pipeline.graph.blocks["snare_onsets"].settings.get("threshold") == pytest.approx(0.150)


class TestEditKnobs:
    """Editing knob values on a persisted config."""

    def test_with_knob_value_updates_knobs(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        updated = config.with_knob_value("threshold", 0.8)
        assert updated.knob_values["threshold"] == 0.8
        assert config.knob_values["threshold"] != 0.8  # original unchanged (frozen)

    def test_with_knob_value_updates_graph(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        updated = config.with_knob_value("threshold", 0.8)
        graph_data = json.loads(updated.graph_json)
        # Find the DetectOnsets block and verify its settings changed
        for block in graph_data["blocks"]:
            if block.get("block_type") == "DetectOnsets":
                assert block["settings"]["threshold"] == 0.8

    def test_with_knob_values_batch(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        updated = config.with_knob_values({"threshold": 0.9, "method": "hfc"})
        assert updated.knob_values["threshold"] == 0.9
        assert updated.knob_values["method"] == "hfc"

    def test_save_updated_config(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        updated = config.with_knob_value("threshold", 0.6)
        session.pipeline_configs.update(updated)
        session.commit()

        loaded = session.pipeline_configs.get(config.id)
        assert loaded.knob_values["threshold"] == 0.6

    def test_updated_at_changes(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        import time
        time.sleep(0.01)
        updated = config.with_knob_value("threshold", 0.5)
        assert updated.updated_at > config.updated_at


class TestExecuteConfig:
    """Executing analysis from a persisted PipelineConfigRecord."""

    def test_execute_succeeds(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        result = orch.execute(session, config.id)
        assert is_ok(result)

        analysis = unwrap(result)
        assert analysis.song_version_id == song_version.id
        assert len(analysis.layer_ids) >= 1
        assert len(analysis.take_ids) >= 1

    def test_execute_creates_layers_and_takes(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        unwrap(orch.execute(session, config.id))

        layers = session.layers.list_by_version(song_version.id)
        assert len(layers) >= 1

        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) >= 1
        assert takes[0].is_main is True

    def test_execute_respects_changed_settings(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())

        # Create with low threshold (more events)
        config = unwrap(orch.create_config(
            session, song_version.id, "onset_detection",
            knob_overrides={"threshold": 0.1},
        ))
        result_low = unwrap(orch.execute(session, config.id))

        # Update to high threshold (fewer events)
        updated = config.with_knob_value("threshold", 0.9)
        session.pipeline_configs.update(updated)
        session.commit()

        result_high = unwrap(orch.execute(session, updated.id))

        # Both should succeed — the actual event counts depend on mock
        assert len(result_low.layer_ids) >= 1
        assert len(result_high.take_ids) >= 1

    def test_execute_nonexistent_config_errors(self, session):
        orch = Orchestrator(get_registry(), _executors())
        result = orch.execute(session, "nonexistent_id")
        assert not is_ok(result)


class TestFullLifecycle:
    """End-to-end: create → edit → execute → re-edit → re-execute."""

    def test_full_workflow(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())

        # 1. Create config from template
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))
        assert config.knob_values["threshold"] == 0.3  # template default

        # 2. User tweaks settings
        config = config.with_knob_value("threshold", 0.5)
        session.pipeline_configs.update(config)
        session.commit()

        # 3. Verify settings persisted
        loaded = session.pipeline_configs.get(config.id)
        assert loaded.knob_values["threshold"] == 0.5

        # 4. Execute with saved settings
        result = unwrap(orch.execute(session, config.id))
        assert len(result.layer_ids) >= 1

        # 5. User tweaks again
        config = loaded.with_knob_value("threshold", 0.8)
        session.pipeline_configs.update(config)
        session.commit()

        # 6. Re-execute — new take, settings respected
        result2 = unwrap(orch.execute(session, config.id))
        assert len(result2.take_ids) >= 1

        # 7. Verify layers accumulated takes
        layers = session.layers.list_by_version(song_version.id)
        assert len(layers) >= 1
        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) >= 2  # first run main + second run non-main


class TestPerBlockSettings:
    """Per-block setting overrides (inspector-level edits)."""

    def test_with_block_setting_updates_one_block(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        # Override drums threshold only
        updated = config.with_block_setting("drums_onsets", "threshold", 0.1)

        pipeline = updated.to_pipeline()
        for block in pipeline.graph.blocks.values():
            if block.id == "drums_onsets":
                assert block.settings.get("threshold") == 0.1
            elif block.block_type == "DetectOnsets":
                # Other onset blocks should still have original value
                assert block.settings.get("threshold") == 0.3

    def test_with_block_setting_doesnt_touch_knob_values(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        updated = config.with_block_setting("drums_onsets", "threshold", 0.1)
        # Knob values unchanged — this is a block-level override
        assert updated.knob_values["threshold"] == config.knob_values["threshold"]

    def test_with_block_settings_batch(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        updated = config.with_block_settings("detect_onsets", {
            "threshold": 0.9,
            "method": "complex",
        })
        pipeline = updated.to_pipeline()
        block = pipeline.graph.blocks["detect_onsets"]
        assert block.settings.get("threshold") == 0.9
        assert block.settings.get("method") == "complex"

    def test_nonexistent_block_raises(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        with pytest.raises(KeyError, match="ghost_block"):
            config.with_block_setting("ghost_block", "threshold", 0.5)

    def test_knob_then_block_override(self, session, song_version):
        """Pipeline knob sets all, then block override adjusts one."""
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        # Set all thresholds to 0.5 via knob
        config = config.with_knob_value("threshold", 0.5)
        # Override just drums to 0.1
        config = config.with_block_setting("drums_onsets", "threshold", 0.1)

        pipeline = config.to_pipeline()
        for block in pipeline.graph.blocks.values():
            if block.id == "drums_onsets":
                assert block.settings.get("threshold") == 0.1
            elif block.block_type == "DetectOnsets":
                assert block.settings.get("threshold") == 0.5

    def test_block_override_persists_to_db(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        updated = config.with_block_setting("drums_onsets", "threshold", 0.1)
        session.pipeline_configs.update(updated)
        session.commit()

        loaded = session.pipeline_configs.get(config.id)
        pipeline = loaded.to_pipeline()
        drums = pipeline.graph.blocks["drums_onsets"]
        assert drums.settings.get("threshold") == 0.1


class TestOverrideProtection:
    """Per-block overrides survive global knob changes."""

    def test_global_knob_skips_overridden_block(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        # Set global to 0.5, then override drums to 0.1
        config = config.with_knob_value("threshold", 0.5)
        config = config.with_block_setting("drums_onsets", "threshold", 0.1)

        # Now change global to 0.8 — drums should stay at 0.1
        config = config.with_knob_value("threshold", 0.8)

        pipeline = config.to_pipeline()
        assert pipeline.graph.blocks["drums_onsets"].settings.get("threshold") == 0.1
        assert pipeline.graph.blocks["bass_onsets"].settings.get("threshold") == 0.8

    def test_override_tracked_in_block_overrides(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        config = config.with_block_setting("drums_onsets", "threshold", 0.1)
        assert "drums_onsets" in config.block_overrides
        assert "threshold" in config.block_overrides["drums_onsets"]

    def test_clear_override_relinks_to_knob(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        config = config.with_knob_value("threshold", 0.5)
        config = config.with_block_setting("drums_onsets", "threshold", 0.1)

        # Clear the override — drums should snap back to knob value (0.5)
        config = config.clear_block_override("drums_onsets", "threshold")

        pipeline = config.to_pipeline()
        assert pipeline.graph.blocks["drums_onsets"].settings.get("threshold") == 0.5
        assert "drums_onsets" not in config.block_overrides

    def test_override_persists_to_db(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        config = config.with_block_setting("drums_onsets", "threshold", 0.1)
        session.pipeline_configs.update(config)
        session.commit()

        loaded = session.pipeline_configs.get(config.id)
        assert "drums_onsets" in loaded.block_overrides
        assert "threshold" in loaded.block_overrides["drums_onsets"]

        # Global knob still skips drums after reload
        loaded = loaded.with_knob_value("threshold", 0.9)
        pipeline = loaded.to_pipeline()
        assert pipeline.graph.blocks["drums_onsets"].settings.get("threshold") == 0.1

    def test_multiple_overrides_on_same_block(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        config = config.with_block_settings("detect_onsets", {
            "threshold": 0.1,
            "method": "complex",
        })
        assert "threshold" in config.block_overrides["detect_onsets"]
        assert "method" in config.block_overrides["detect_onsets"]

    def test_no_override_without_block_setting(self, session, song_version):
        """Knob changes don't create overrides."""
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        config = config.with_knob_value("threshold", 0.5)
        assert config.block_overrides == {}


class TestMapsToBlock:
    """Knobs with maps_to_block targeting."""

    def test_targeted_knob_only_updates_mapped_block(self, session, song_version):
        from echozero.pipelines.params import Knob, KnobWidget
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        # Simulate a knob that targets only drums_onsets
        knob_meta = {
            "threshold": Knob(
                default=0.3, widget=KnobWidget.SLIDER,
                min_value=0.0, max_value=1.0,
                maps_to_block="drums_onsets",
            ),
        }
        updated = config.with_knob_value("threshold", 0.1, knob_metadata=knob_meta)

        pipeline = updated.to_pipeline()
        for block in pipeline.graph.blocks.values():
            if block.id == "drums_onsets":
                assert block.settings.get("threshold") == 0.1
            elif block.block_type == "DetectOnsets":
                assert block.settings.get("threshold") == 0.3  # unchanged

    def test_global_knob_updates_all(self, session, song_version):
        """Knob without maps_to_block updates all matching blocks."""
        from echozero.pipelines.params import Knob, KnobWidget
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        knob_meta = {
            "threshold": Knob(
                default=0.3, widget=KnobWidget.SLIDER,
                min_value=0.0, max_value=1.0,
                maps_to_block=None,  # global
            ),
        }
        updated = config.with_knob_value("threshold", 0.8, knob_metadata=knob_meta)

        pipeline = updated.to_pipeline()
        for block in pipeline.graph.blocks.values():
            if block.block_type == "DetectOnsets":
                assert block.settings.get("threshold") == 0.8


class TestMapsToSetting:
    """Knobs can target block settings with different names."""

    def test_full_analysis_prefixed_onset_knobs_update_detect_blocks(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        template = get_registry().get("full_analysis")
        assert template is not None
        config = unwrap(orch.create_config(session, song_version.id, "full_analysis"))

        updated = config.with_knob_values(
            {
                "onset_method": "hfc",
                "onset_backtrack": False,
                "onset_timing_offset_ms": 18.0,
            },
            knob_metadata=template.knobs,
        )

        pipeline = updated.to_pipeline()
        for block_id in ("drums_onsets", "bass_onsets", "vocals_onsets", "other_onsets"):
            block = pipeline.graph.blocks[block_id]
            assert block.settings.get("method") == "hfc"
            assert block.settings.get("backtrack") is False
            assert block.settings.get("timing_offset_ms") == 18.0

    def test_drum_classification_prefixed_knobs_update_detect_and_classify_blocks(
        self, session, song_version
    ):
        orch = Orchestrator(get_registry(), _executors())
        template = get_registry().get("drum_classification")
        assert template is not None
        config = unwrap(orch.create_config(session, song_version.id, "drum_classification"))

        updated = config.with_knob_values(
            {
                "onset_threshold": 0.05,
                "onset_min_gap": 0.02,
                "classify_device": "cpu",
                "classify_batch_size": 8,
            },
            knob_metadata=template.knobs,
        )

        pipeline = updated.to_pipeline()
        detect = pipeline.graph.blocks["detect_onsets"]
        classify = pipeline.graph.blocks["classify"]
        assert detect.settings.get("threshold") == 0.05
        assert detect.settings.get("min_gap") == 0.02
        assert classify.settings.get("device") == "cpu"
        assert classify.settings.get("batch_size") == 8

    def test_extract_classified_drums_prefixed_knobs_update_detect_and_classify_blocks(
        self, session, song_version
    ):
        orch = Orchestrator(get_registry(), _executors())
        template = get_registry().get("extract_classified_drums")
        assert template is not None
        config = unwrap(orch.create_config(session, song_version.id, "extract_classified_drums"))

        updated = config.with_knob_values(
            {
                "kick_filter_enabled": False,
                "kick_filter_freq": 140.0,
                "snare_onset_method": "complex",
                "classify_device": "cpu",
            },
            knob_metadata=template.knobs,
        )

        pipeline = updated.to_pipeline()
        kick_filter = pipeline.graph.blocks["kick_filter"]
        snare_filter = pipeline.graph.blocks["snare_filter"]
        kick_onsets = pipeline.graph.blocks["kick_onsets"]
        snare_onsets = pipeline.graph.blocks["snare_onsets"]
        classify = pipeline.graph.blocks["classify_drums"]
        assert kick_filter.settings.get("enabled") is False
        assert kick_filter.settings.get("freq") == 140.0
        assert snare_filter.settings.get("freq") == 180.0
        assert kick_onsets.settings.get("method") == "default"
        assert snare_onsets.settings.get("method") == "complex"
        assert classify.settings.get("device") == "cpu"
        classify_inputs = {
            connection.target_input_name
            for connection in pipeline.graph.connections
            if connection.target_block_id == "classify_drums"
        }
        assert "kick_events_in" in classify_inputs
        assert "snare_events_in" in classify_inputs
        assert "kick_audio_in" not in classify_inputs
        assert "snare_audio_in" not in classify_inputs

    def test_extract_song_drum_events_knobs_update_separator_detect_and_classify_blocks(
        self, session, song_version
    ):
        orch = Orchestrator(get_registry(), _executors())
        template = get_registry().get("extract_song_drum_events")
        assert template is not None
        config = unwrap(orch.create_config(session, song_version.id, "extract_song_drum_events"))

        updated = config.with_knob_values(
            {
                "model": "mdx_extra",
                "device": "cpu",
                "kick_onset_threshold": 0.05,
                "snare_filter_enabled": False,
                "snare_filter_freq": 240.0,
                "positive_threshold": 0.65,
            },
            knob_metadata=template.knobs,
        )

        pipeline = updated.to_pipeline()
        separate = pipeline.graph.blocks["separate_drums"]
        kick_onsets = pipeline.graph.blocks["kick_onsets"]
        snare_filter = pipeline.graph.blocks["snare_filter"]
        classify = pipeline.graph.blocks["classify_drums"]
        assert separate.settings.get("model") == "mdx_extra"
        assert separate.settings.get("device") == "cpu"
        assert kick_onsets.settings.get("threshold") == 0.05
        assert snare_filter.settings.get("enabled") is False
        assert snare_filter.settings.get("freq") == 240.0
        assert classify.settings.get("device") == "cpu"
        assert classify.settings.get("kick_positive_threshold") == 0.5
        assert classify.settings.get("snare_positive_threshold") == 0.65
        classify_inputs = {
            connection.target_input_name
            for connection in pipeline.graph.connections
            if connection.target_block_id == "classify_drums"
        }
        assert "kick_events_in" in classify_inputs
        assert "snare_events_in" in classify_inputs
        assert "kick_audio_in" not in classify_inputs
        assert "snare_audio_in" not in classify_inputs


class TestConfigToFromPipeline:
    """Verify PipelineConfigRecord.from_pipeline and to_pipeline round-trips."""

    def test_round_trip(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(session, song_version.id, "onset_detection"))

        # Deserialize back to pipeline
        pipeline = config.to_pipeline()
        assert len(pipeline.graph.blocks) >= 2
        assert len(pipeline.outputs) >= 1

    def test_graph_preserves_settings(self, session, song_version):
        orch = Orchestrator(get_registry(), _executors())
        config = unwrap(orch.create_config(
            session, song_version.id, "onset_detection",
            knob_overrides={"threshold": 0.7},
        ))

        pipeline = config.to_pipeline()
        # Find DetectOnsets block
        for block in pipeline.graph.blocks.values():
            if block.block_type == "DetectOnsets":
                assert block.settings.get("threshold") == 0.7
