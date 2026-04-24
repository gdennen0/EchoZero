"""Application service for machine-local EchoZero preferences.
Exists to keep app settings typed, validated, and reusable outside any Qt surface.
Connects local settings storage to launcher/runtime configuration and neutral settings pages.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from echozero.application.settings.contracts import SettingsOption, SettingsPage
from echozero.application.settings.models import (
    AppPreferences,
    AppSettingsLaunchOverrides,
    AppSettingsUpdateResult,
    AudioLatencyProfile,
    AudioOutputPreferences,
    AudioOutputRuntimeConfig,
    MA3OscPreferences,
    MA3OscRuntimeConfig,
    OscReceivePreferences,
    OscReceiveRuntimeConfig,
    OscSendPreferences,
    OscSendRuntimeConfig,
)
from echozero.application.settings.page_builder import (
    build_app_settings_page,
    list_audio_output_device_options,
)


class AppSettingsValidationError(ValueError):
    """Raised when app settings values do not pass validation."""


class AppSettingsStore(Protocol):
    """Storage protocol for machine-local app preferences."""

    path: Path

    def load(self) -> AppPreferences: ...

    def save(self, preferences: AppPreferences) -> None: ...


class AppSettingsService:
    """Own app-preferences loading, validation, settings-page rendering, and resolution."""

    def __init__(
        self,
        store: AppSettingsStore,
        *,
        audio_device_options_provider: Callable[[], tuple[SettingsOption, ...]] = (
            list_audio_output_device_options
        ),
    ) -> None:
        self._store = store
        self._audio_device_options_provider = audio_device_options_provider
        self._preferences = store.load()

    @property
    def store_path(self) -> Path:
        """Filesystem location backing the current settings store."""

        return self._store.path

    def preferences(self) -> AppPreferences:
        """Return the currently loaded app preferences."""

        return self._preferences

    def reload(self) -> AppPreferences:
        """Reload app preferences from storage."""

        self._preferences = self._store.load()
        return self._preferences

    def describe(self, *, include_hidden: bool = False) -> SettingsPage:
        """Render the current app preferences into a neutral settings page."""

        return build_app_settings_page(
            self._preferences,
            audio_device_options_provider=self._audio_device_options_provider,
            include_hidden=include_hidden,
        )

    def default_values(self) -> dict[str, object]:
        """Return dialog-ready default values for the current settings page."""

        return self._values_from_preferences(AppPreferences())

    def restore_defaults(self) -> AppSettingsUpdateResult:
        """Persist default app preferences and report the change impact."""

        return self.replace_preferences(AppPreferences())

    def apply_updates(self, updates: Mapping[str, object]) -> AppSettingsUpdateResult:
        """Validate and persist one partial field update set."""

        current = self._preferences
        next_preferences = AppPreferences(
            audio_output=self._updated_audio_preferences(
                current.audio_output,
                updates,
            ),
            ma3_osc=self._updated_ma3_osc_preferences(
                current.ma3_osc,
                updates,
            ),
        )
        return self.replace_preferences(next_preferences)

    def replace_preferences(self, preferences: AppPreferences) -> AppSettingsUpdateResult:
        """Validate and persist one full app-preferences snapshot."""

        current = self._preferences
        self._validate(preferences)
        self._store.save(preferences)
        self._preferences = preferences

        audio_changed = current.audio_output != preferences.audio_output
        osc_changed = current.ma3_osc != preferences.ma3_osc
        restart_reasons = self._restart_reasons(
            audio_changed=audio_changed,
            osc_changed=osc_changed,
        )
        return AppSettingsUpdateResult(
            preferences=preferences,
            audio_changed=audio_changed,
            osc_changed=osc_changed,
            restart_required=bool(restart_reasons),
            restart_reasons=restart_reasons,
        )

    def resolve_audio_output_config(self) -> AudioOutputRuntimeConfig:
        """Resolve saved audio preferences into one runtime config."""

        audio = self._preferences.audio_output
        return AudioOutputRuntimeConfig(
            output_device=self._runtime_output_device(audio.output_device),
            sample_rate=audio.sample_rate,
            channels=audio.output_channels,
            stream_latency=(
                None
                if audio.latency_profile is AudioLatencyProfile.AUTO
                else audio.latency_profile.value
            ),
            stream_blocksize=audio.blocksize,
            prime_output_buffers_using_stream_callback=audio.prime_output_buffers_using_stream_callback,
        )

    def resolve_ma3_osc_runtime_config(
        self,
        *,
        launch_overrides: AppSettingsLaunchOverrides | None = None,
    ) -> MA3OscRuntimeConfig:
        """Resolve saved and launch-override MA3 OSC settings into one runtime config."""

        overrides = launch_overrides or AppSettingsLaunchOverrides()
        osc = self._preferences.ma3_osc

        receive_enabled = (
            osc.receive.enabled
            or overrides.ma3_osc_listen_port is not None
        )
        send_enabled = (
            osc.send.enabled
            or overrides.ma3_osc_command_port is not None
        )
        return MA3OscRuntimeConfig(
            receive=OscReceiveRuntimeConfig(
                enabled=receive_enabled,
                host=overrides.ma3_osc_listen_host or osc.receive.host,
                port=(
                    overrides.ma3_osc_listen_port
                    if overrides.ma3_osc_listen_port is not None
                    else osc.receive.port
                ),
            ),
            send=OscSendRuntimeConfig(
                enabled=send_enabled,
                host=overrides.ma3_osc_command_host or osc.send.host,
                port=(
                    overrides.ma3_osc_command_port
                    if overrides.ma3_osc_command_port is not None
                    else osc.send.port
                ),
            ),
        )

    def _updated_audio_preferences(
        self,
        current: AudioOutputPreferences,
        updates: Mapping[str, object],
    ) -> AudioOutputPreferences:
        return AudioOutputPreferences(
            output_device=self._device_value(
                updates.get("audio.output_device"),
                current.output_device,
            ),
            sample_rate=self._optional_positive_int(
                updates.get("audio.sample_rate"),
                current.sample_rate,
            ),
            output_channels=self._optional_positive_int(
                updates.get("audio.output_channels"),
                current.output_channels,
            ),
            latency_profile=self._latency_profile(
                updates.get("audio.latency_profile"),
                current.latency_profile,
            ),
            blocksize=self._optional_positive_int(
                updates.get("audio.blocksize"),
                current.blocksize,
            ),
            prime_output_buffers_using_stream_callback=bool(
                updates.get(
                    "audio.prime_output_buffers_using_stream_callback",
                    current.prime_output_buffers_using_stream_callback,
                )
            ),
        )

    def _updated_ma3_osc_preferences(
        self,
        current: MA3OscPreferences,
        updates: Mapping[str, object],
    ) -> MA3OscPreferences:
        return MA3OscPreferences(
            receive=OscReceivePreferences(
                enabled=bool(updates.get("osc_receive.enabled", current.receive.enabled)),
                host=self._text(updates.get("osc_receive.host"), current.receive.host),
                port=self._non_negative_int(
                    updates.get("osc_receive.port"),
                    current.receive.port,
                ),
            ),
            send=OscSendPreferences(
                enabled=bool(updates.get("osc_send.enabled", current.send.enabled)),
                host=self._text(updates.get("osc_send.host"), current.send.host),
                port=self._optional_positive_int(
                    updates.get("osc_send.port"),
                    current.send.port,
                ),
            ),
        )

    def _validate(self, preferences: AppPreferences) -> None:
        audio = preferences.audio_output
        if audio.sample_rate is not None and audio.sample_rate <= 0:
            raise AppSettingsValidationError("Audio sample rate override must be greater than 0.")
        if audio.output_channels is not None and audio.output_channels not in {1, 2}:
            raise AppSettingsValidationError("Audio output channels must be 1, 2, or Auto.")
        if audio.blocksize is not None and audio.blocksize <= 0:
            raise AppSettingsValidationError("Audio blocksize override must be greater than 0.")

        receive = preferences.ma3_osc.receive
        if receive.enabled and not receive.host.strip():
            raise AppSettingsValidationError("OSC receive bind address is required when receive is enabled.")
        if not 0 <= receive.port <= 65535:
            raise AppSettingsValidationError("OSC receive bind port must be between 0 and 65535.")

        send = preferences.ma3_osc.send
        if send.enabled:
            if not send.host.strip():
                raise AppSettingsValidationError(
                    "OSC send target address is required when send is enabled."
                )
            if send.port is None or not (1 <= send.port <= 65535):
                raise AppSettingsValidationError(
                    "OSC send target port must be between 1 and 65535 when send is enabled."
                )
        if send.port is not None and not (1 <= send.port <= 65535):
            raise AppSettingsValidationError("OSC send target port must be between 1 and 65535.")
    @staticmethod
    def _values_from_preferences(preferences: AppPreferences) -> dict[str, object]:
        return {
            "audio.output_device": preferences.audio_output.output_device or "",
            "audio.sample_rate": preferences.audio_output.sample_rate or 0,
            "audio.output_channels": preferences.audio_output.output_channels or 0,
            "audio.latency_profile": preferences.audio_output.latency_profile.value,
            "audio.blocksize": preferences.audio_output.blocksize or 0,
            "audio.prime_output_buffers_using_stream_callback": (
                preferences.audio_output.prime_output_buffers_using_stream_callback
            ),
            "osc_receive.enabled": preferences.ma3_osc.receive.enabled,
            "osc_receive.host": preferences.ma3_osc.receive.host,
            "osc_receive.port": preferences.ma3_osc.receive.port,
            "osc_send.enabled": preferences.ma3_osc.send.enabled,
            "osc_send.host": preferences.ma3_osc.send.host,
            "osc_send.port": preferences.ma3_osc.send.port or 0,
        }

    @staticmethod
    def _restart_reasons(*, audio_changed: bool, osc_changed: bool) -> tuple[str, ...]:
        reasons: list[str] = []
        if audio_changed:
            reasons.append("Restart EchoZero to apply saved audio output settings.")
        if osc_changed:
            reasons.append("Restart EchoZero to apply saved OSC settings.")
        return tuple(reasons)

    @staticmethod
    def _runtime_output_device(value: str | None) -> int | str | None:
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        return text

    @staticmethod
    def _latency_profile(value: object, current: AudioLatencyProfile) -> AudioLatencyProfile:
        try:
            return AudioLatencyProfile(str(value or current.value).strip().lower())
        except ValueError:
            return current

    @staticmethod
    def _text(value: object, current: str) -> str:
        text = str(value if value is not None else current).strip()
        return text or current

    @staticmethod
    def _device_value(value: object, current: str | None) -> str | None:
        text = str(value if value is not None else (current or "")).strip()
        return text or None

    @staticmethod
    def _non_negative_int(value: object, current: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return current

    @staticmethod
    def _optional_positive_int(value: object, current: int | None) -> int | None:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return current
        return resolved if resolved > 0 else None


def build_default_app_settings_service(path: Path | None = None) -> AppSettingsService:
    """Build the canonical app-settings service backed by the local JSON store."""

    from echozero.infrastructure.settings.json_store import JsonAppSettingsStore

    return AppSettingsService(JsonAppSettingsStore(path=path))
