"""Behavior-oriented app-settings service tests.
Exists to keep machine-local settings validation and resolution covered outside the UI shell.
Connects the reusable app-settings lane to stable in-memory store doubles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from echozero.application.settings import (
    AppPreferences,
    AppSettingsLaunchOverrides,
    AppSettingsService,
    AppSettingsValidationError,
    AudioLatencyProfile,
    AudioOutputPreferences,
    MA3OscPreferences,
    OscReceivePreferences,
    OscSendPreferences,
    SettingsFieldSurface,
    SettingsFieldWidget,
    SettingsOption,
)
from echozero.application.settings.models import (
    app_preferences_from_dict,
    import_safe_pipeline_action_descriptors,
)


class _MemoryStore:
    """In-memory app-settings store for service tests."""

    path = Path("/tmp/echozero-test-app-settings.json")

    def __init__(self, preferences: AppPreferences | None = None) -> None:
        self._preferences = preferences or AppPreferences()

    def load(self) -> AppPreferences:
        return self._preferences

    def save(self, preferences: AppPreferences) -> None:
        self._preferences = preferences


def _device_options() -> tuple[SettingsOption, ...]:
    return (
        SettingsOption(value="", label="System Default", metadata={"max_output_channels": 2}),
        SettingsOption(value="7", label="Studio Output", metadata={"max_output_channels": 8}),
    )


def test_app_settings_service_describe_surfaces_audio_osc_and_import_sections() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    page = service.describe()
    fields = tuple(field for section in page.sections for field in section.fields)
    import_toggle_keys = {
        f"import.pipeline_action.{descriptor.action_id}"
        for descriptor in import_safe_pipeline_action_descriptors()
    }

    assert [section.key for section in page.sections] == [
        "audio",
        "osc_receive",
        "osc_send",
        "song_import",
    ]
    assert any(
        field.key == "audio.blocksize" and field.surface is SettingsFieldSurface.ADVANCED
        for field in fields
    )
    assert page.sections[0].preferred_columns == 2
    assert page.sections[0].always_show_advanced is True
    assert any(field.key == "osc_receive.port" for field in fields)
    assert any(field.key == "osc_send.port" for field in fields)
    assert any(field.key == "import.strip_ltc_timecode" for field in fields)
    assert import_toggle_keys
    assert import_toggle_keys <= {field.key for field in fields}
    output_channels_field = next(field for field in fields if field.key == "audio.output_channels")
    output_channel_values = {int(option.value) for option in output_channels_field.options}
    assert 0 in output_channel_values
    assert 16 in output_channel_values
    master_bus_field = next(field for field in fields if field.key == "audio.master_output_bus")
    assert master_bus_field.widget is SettingsFieldWidget.CHECKBOX_GROUP
    assert [(option.value, option.label) for option in master_bus_field.options] == [
        ("outputs_1_1", "Output 1"),
        ("outputs_2_2", "Output 2"),
    ]
    assert page.warnings == (
        "Changes are saved to the local config JSON only.",
        "Changes apply live in the active shell when supported, and always apply on next launch.",
    )


def test_app_settings_service_describe_uses_selected_device_channels_for_master_bus_options() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    page = service.describe_with_updates({"audio.output_device": "7"})
    fields = tuple(field for section in page.sections for field in section.fields)
    master_bus_field = next(field for field in fields if field.key == "audio.master_output_bus")

    assert [option.value for option in master_bus_field.options] == [
        "outputs_1_1",
        "outputs_2_2",
        "outputs_3_3",
        "outputs_4_4",
        "outputs_5_5",
        "outputs_6_6",
        "outputs_7_7",
        "outputs_8_8",
    ]


def test_app_settings_service_describe_limits_master_bus_options_to_device_channels() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    page = service.describe_with_updates(
        {"audio.output_device": "7", "audio.output_channels": 4}
    )
    fields = tuple(field for section in page.sections for field in section.fields)
    master_bus_field = next(field for field in fields if field.key == "audio.master_output_bus")

    assert [option.value for option in master_bus_field.options] == [
        "outputs_1_1",
        "outputs_2_2",
        "outputs_3_3",
        "outputs_4_4",
        "outputs_5_5",
        "outputs_6_6",
        "outputs_7_7",
        "outputs_8_8",
    ]


def test_app_settings_service_apply_updates_persists_audio_osc_and_import_changes() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    result = service.apply_updates(
        {
            "audio.sample_rate": 48000,
            "audio.output_channels": 2,
            "osc_send.enabled": True,
            "osc_send.port": 9000,
            "import.strip_ltc_timecode": True,
            "import.pipeline_action.timeline.extract_stems": True,
            "import.pipeline_action.timeline.extract_song_drum_events": True,
        }
    )

    assert result.audio_changed is True
    assert result.osc_changed is True
    assert result.song_import_changed is True
    assert result.restart_required is False
    assert result.restart_reasons == ()
    assert result.preferences.audio_output.sample_rate == 48000
    assert result.preferences.ma3_osc.send.enabled is True
    assert result.preferences.ma3_osc.send.port == 9000
    assert result.preferences.song_import.strip_ltc_timecode is True
    assert result.preferences.song_import.pipeline_action_ids == (
        "timeline.extract_stems",
        "timeline.extract_song_drum_events",
    )


def test_app_settings_service_replaces_pipeline_defaults_per_template() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    service.replace_pipeline_defaults(
        "stem_separation",
        {"model": "mdx_extra_q", "device": "cpu"},
    )

    assert service.pipeline_defaults_for_template("stem_separation") == {
        "model": "mdx_extra_q",
        "device": "cpu",
    }


def test_app_settings_service_apply_updates_accepts_four_output_channels() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    result = service.apply_updates({"audio.output_channels": 4})

    assert result.audio_changed is True
    assert result.preferences.audio_output.output_channels == 4


def test_app_settings_service_apply_updates_accepts_mirrored_master_output_channels() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    result = service.apply_updates(
        {
            "audio.output_channels": 4,
            "audio.master_output_bus": "outputs_1_1, outputs_2_2",
        }
    )

    assert result.preferences.audio_output.master_output_bus == "outputs_1_1,outputs_2_2"
    assert service.resolve_audio_output_config().master_output_bus == "outputs_1_1,outputs_2_2"


def test_app_settings_service_prunes_master_outputs_when_channel_count_shrinks() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    result = service.apply_updates(
        {
            "audio.output_channels": 2,
            "audio.master_output_bus": "outputs_1_1,outputs_3_3",
        }
    )

    assert result.preferences.audio_output.master_output_bus == "outputs_1_1"


def test_app_settings_service_prunes_master_outputs_when_selected_device_changes() -> None:
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                audio_output=AudioOutputPreferences(
                    output_device="7",
                    master_output_bus="outputs_1_1,outputs_8_8",
                )
            )
        ),
        audio_device_options_provider=_device_options,
    )

    result = service.apply_updates({"audio.output_device": ""})

    assert result.preferences.audio_output.output_device is None
    assert result.preferences.audio_output.master_output_bus == "outputs_1_1"


def test_app_settings_service_apply_updates_rejects_output_channels_above_supported_range() -> (
    None
):
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    with pytest.raises(AppSettingsValidationError, match="between 1 and 16"):
        service.apply_updates({"audio.output_channels": 17})


def test_app_settings_service_resolve_audio_output_config_converts_runtime_types() -> None:
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                audio_output=AudioOutputPreferences(
                    output_device="7",
                    sample_rate=48000,
                    output_channels=2,
                    latency_profile=AudioLatencyProfile.LOW,
                    blocksize=512,
                    prime_output_buffers_using_stream_callback=False,
                )
            )
        ),
        audio_device_options_provider=_device_options,
    )

    config = service.resolve_audio_output_config()

    assert config.output_device == 7
    assert config.sample_rate == 48000
    assert config.channels == 2
    assert config.stream_latency == "low"
    assert config.stream_blocksize == 512
    assert config.prime_output_buffers_using_stream_callback is False


def test_app_settings_service_resolve_ma3_osc_runtime_config_merges_launch_overrides() -> None:
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                ma3_osc=MA3OscPreferences(
                    receive=OscReceivePreferences(
                        enabled=True,
                        host="127.0.0.1",
                        port=7001,
                    ),
                    send=OscSendPreferences(
                        enabled=False,
                        host="127.0.0.1",
                        port=None,
                    ),
                )
            )
        ),
        audio_device_options_provider=_device_options,
    )

    config = service.resolve_ma3_osc_runtime_config(
        launch_overrides=AppSettingsLaunchOverrides(
            ma3_osc_listen_port=7100,
            ma3_osc_command_host="10.0.0.2",
            ma3_osc_command_port=9000,
        )
    )

    assert config.receive.enabled is True
    assert config.receive.port == 7100
    assert config.send.enabled is True
    assert config.send.host == "10.0.0.2"
    assert config.send.port == 9000


def test_app_settings_service_recent_project_paths_preserve_order_and_limit() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    service.remember_recent_project_path("C:/projects/alpha.ez")
    service.remember_recent_project_path("C:/projects/bravo.ez")
    service.remember_recent_project_path("C:/projects/alpha.ez")

    recent = service.recent_project_paths()

    assert recent == (
        Path("C:/projects/alpha.ez"),
        Path("C:/projects/bravo.ez"),
    )

    for index in range(12):
        service.remember_recent_project_path(f"C:/projects/show-{index}.ez")
    limited = service.recent_project_paths()

    assert len(limited) == 10
    assert limited[0] == Path("C:/projects/show-11.ez")
    assert limited[-1] == Path("C:/projects/show-2.ez")


def test_app_settings_service_forget_recent_project_path_removes_match() -> None:
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                recent_project_paths=(
                    "C:/projects/alpha.ez",
                    "C:/projects/bravo.ez",
                )
            )
        ),
        audio_device_options_provider=_device_options,
    )

    remaining = service.forget_recent_project_path("C:/projects/ALPHA.ez")

    assert remaining == (Path("C:/projects/bravo.ez"),)


def test_app_preferences_from_dict_accepts_recent_project_paths() -> None:
    preferences = app_preferences_from_dict(
        {
            "recent_project_paths": [
                "C:/projects/alpha.ez",
                "",
                "C:/projects/bravo.ez",
            ]
        }
    )

    assert preferences.recent_project_paths == (
        "C:/projects/alpha.ez",
        "C:/projects/bravo.ez",
    )


def test_app_preferences_from_dict_canonicalizes_mirrored_master_output_buses() -> None:
    preferences = app_preferences_from_dict(
        {
            "audio_output": {
                "master_output_bus": "outputs_1_1, outputs_2_2,outputs_1_1",
            }
        }
    )

    assert preferences.audio_output.master_output_bus == "outputs_1_1,outputs_2_2"


def test_app_preferences_from_dict_parses_song_import_pipeline_actions() -> None:
    preferences = app_preferences_from_dict(
        {
            "song_import": {
                "strip_ltc_timecode": False,
                "pipeline_action_ids": [
                    "timeline.extract_stems",
                    "timeline.extract_song_drum_events",
                ],
            }
        }
    )

    assert preferences.song_import.strip_ltc_timecode is False
    assert preferences.song_import.pipeline_action_ids == (
        "timeline.extract_stems",
        "timeline.extract_song_drum_events",
    )


def test_app_preferences_from_dict_filters_non_import_safe_actions() -> None:
    preferences = app_preferences_from_dict(
        {
            "song_import": {
                "pipeline_action_ids": [
                    "timeline.extract_stems",
                    "timeline.classify_drum_events",
                ]
            }
        }
    )

    assert preferences.song_import.pipeline_action_ids == ("timeline.extract_stems",)


def test_app_preferences_from_dict_parses_pipeline_defaults_by_template() -> None:
    preferences = app_preferences_from_dict(
        {
            "pipeline_defaults_by_template": {
                "stem_separation": {"model": "mdx_extra_q", "device": "cpu"},
                "": {"ignored": True},
            }
        }
    )

    assert preferences.pipeline_defaults_by_template == {
        "stem_separation": {"model": "mdx_extra_q", "device": "cpu"}
    }
