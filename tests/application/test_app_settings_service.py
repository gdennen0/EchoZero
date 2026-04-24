"""Behavior-oriented app-settings service tests.
Exists to keep machine-local settings validation and resolution covered outside the UI shell.
Connects the reusable app-settings lane to stable in-memory store doubles.
"""

from __future__ import annotations

from pathlib import Path

from echozero.application.settings import (
    AppPreferences,
    AppSettingsLaunchOverrides,
    AppSettingsService,
    AudioLatencyProfile,
    AudioOutputPreferences,
    MA3OscPreferences,
    OscReceivePreferences,
    OscSendPreferences,
    SettingsFieldSurface,
    SettingsOption,
)
from echozero.application.settings.models import app_preferences_from_dict


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
        SettingsOption(value="", label="System Default"),
        SettingsOption(value="7", label="Studio Output"),
    )


def test_app_settings_service_describe_surfaces_audio_and_osc_sections() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    page = service.describe()
    fields = tuple(field for section in page.sections for field in section.fields)

    assert [section.key for section in page.sections] == [
        "audio",
        "osc_receive",
        "osc_send",
    ]
    assert any(
        field.key == "audio.blocksize" and field.surface is SettingsFieldSurface.ADVANCED
        for field in fields
    )
    assert any(field.key == "osc_receive.port" for field in fields)
    assert any(field.key == "osc_send.port" for field in fields)
    assert page.warnings == (
        "Changes are saved to the local config JSON only.",
        "Restart EchoZero to use updated audio or OSC settings.",
    )


def test_app_settings_service_apply_updates_persists_audio_and_osc_changes() -> None:
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)

    result = service.apply_updates(
        {
            "audio.sample_rate": 48000,
            "audio.output_channels": 2,
            "osc_send.enabled": True,
            "osc_send.port": 9000,
        }
    )

    assert result.audio_changed is True
    assert result.osc_changed is True
    assert result.restart_required is True
    assert result.restart_reasons == (
        "Restart EchoZero to apply saved audio output settings.",
        "Restart EchoZero to apply saved OSC settings.",
    )
    assert result.preferences.audio_output.sample_rate == 48000
    assert result.preferences.ma3_osc.send.enabled is True
    assert result.preferences.ma3_osc.send.port == 9000


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


def test_app_preferences_from_dict_accepts_legacy_flat_ma3_shape() -> None:
    preferences = app_preferences_from_dict(
        {
            "ma3_osc": {
                "listen_enabled": True,
                "listen_host": "127.0.0.1",
                "listen_port": 7100,
                "command_enabled": True,
                "command_host": "10.0.0.2",
                "command_port": 9000,
            }
        }
    )

    assert preferences.ma3_osc.receive == OscReceivePreferences(
        enabled=True,
        host="127.0.0.1",
        port=7100,
    )
    assert preferences.ma3_osc.send == OscSendPreferences(
        enabled=True,
        host="10.0.0.2",
        port=9000,
    )
