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
    SongImportNameMode,
    SongImportPreferences,
    canonical_import_pipeline_action_ids,
    import_safe_pipeline_action_descriptors,
)
from echozero.application.settings.page_builder import (
    build_app_settings_page,
    list_audio_output_device_options,
)
from echozero.application.settings.network_options import list_osc_receive_address_options
from echozero.output_routing import (
    DEFAULT_MASTER_OUTPUT_BUS,
    canonical_master_output_buses,
    parse_output_bus_spans,
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
        osc_receive_address_options_provider: Callable[[], tuple[SettingsOption, ...]] = (
            list_osc_receive_address_options
        ),
    ) -> None:
        self._store = store
        self._audio_device_options_provider = audio_device_options_provider
        self._osc_receive_address_options_provider = osc_receive_address_options_provider
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
            osc_receive_address_options_provider=self._osc_receive_address_options_provider,
            include_hidden=include_hidden,
        )

    def describe_with_updates(
        self,
        updates: Mapping[str, object],
        *,
        include_hidden: bool = False,
    ) -> SettingsPage:
        """Render the settings page as if one unsaved form snapshot were applied."""

        current = self._preferences
        draft_preferences = AppPreferences(
            audio_output=self._updated_audio_preferences(current.audio_output, updates),
            ma3_osc=self._updated_ma3_osc_preferences(current.ma3_osc, updates),
            song_import=self._updated_song_import_preferences(current.song_import, updates),
            pipeline_defaults_by_template=current.pipeline_defaults_by_template,
            pipeline_profiles_by_template=current.pipeline_profiles_by_template,
            recent_project_paths=current.recent_project_paths,
        )
        return build_app_settings_page(
            draft_preferences,
            audio_device_options_provider=self._audio_device_options_provider,
            osc_receive_address_options_provider=self._osc_receive_address_options_provider,
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
            song_import=self._updated_song_import_preferences(
                current.song_import,
                updates,
            ),
            pipeline_defaults_by_template=current.pipeline_defaults_by_template,
            pipeline_profiles_by_template=current.pipeline_profiles_by_template,
            recent_project_paths=current.recent_project_paths,
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
        song_import_changed = current.song_import != preferences.song_import
        return AppSettingsUpdateResult(
            preferences=preferences,
            audio_changed=audio_changed,
            osc_changed=osc_changed,
            song_import_changed=song_import_changed,
        )

    def pipeline_defaults_for_template(self, template_id: str) -> dict[str, object]:
        """Return saved machine-local pipeline defaults for one template."""

        return dict(self._preferences.pipeline_defaults_by_template.get(template_id, {}))

    def replace_pipeline_defaults(
        self,
        template_id: str,
        values: Mapping[str, object],
    ) -> AppSettingsUpdateResult:
        """Persist machine-local pipeline defaults for one template."""

        text = str(template_id).strip()
        if not text:
            raise AppSettingsValidationError("Pipeline defaults require a template id.")
        updated_defaults = {
            key: dict(value)
            for key, value in self._preferences.pipeline_defaults_by_template.items()
        }
        updated_defaults[text] = {
            str(key).strip(): value for key, value in values.items() if str(key).strip()
        }
        return self.replace_preferences(
            AppPreferences(
                audio_output=self._preferences.audio_output,
                ma3_osc=self._preferences.ma3_osc,
                song_import=self._preferences.song_import,
                pipeline_defaults_by_template=updated_defaults,
                pipeline_profiles_by_template=self._preferences.pipeline_profiles_by_template,
                recent_project_paths=self._preferences.recent_project_paths,
            )
        )

    def pipeline_profiles_for_template(
        self,
        template_id: str,
    ) -> dict[str, dict[str, object]]:
        """Return saved machine-local pipeline profiles for one template."""

        return {
            name: dict(values)
            for name, values in self._preferences.pipeline_profiles_by_template.get(
                template_id,
                {},
            ).items()
        }

    def save_pipeline_profile(
        self,
        template_id: str,
        profile_name: str,
        values: Mapping[str, object],
    ) -> AppSettingsUpdateResult:
        """Persist one named machine-local pipeline profile for one template."""

        text = str(template_id).strip()
        if not text:
            raise AppSettingsValidationError("Pipeline profiles require a template id.")
        name = str(profile_name).strip()
        if not name:
            raise AppSettingsValidationError("Pipeline profiles require a profile name.")
        updated_profiles = {
            key: {profile: dict(profile_values) for profile, profile_values in template.items()}
            for key, template in self._preferences.pipeline_profiles_by_template.items()
        }
        template_profiles = dict(updated_profiles.get(text, {}))
        template_profiles[name] = {
            str(key).strip(): value for key, value in values.items() if str(key).strip()
        }
        updated_profiles[text] = template_profiles
        return self.replace_preferences(
            AppPreferences(
                audio_output=self._preferences.audio_output,
                ma3_osc=self._preferences.ma3_osc,
                song_import=self._preferences.song_import,
                pipeline_defaults_by_template=self._preferences.pipeline_defaults_by_template,
                pipeline_profiles_by_template=updated_profiles,
                recent_project_paths=self._preferences.recent_project_paths,
            )
        )

    def resolve_audio_output_config(self) -> AudioOutputRuntimeConfig:
        """Resolve saved audio preferences into one runtime config."""

        audio = self._preferences.audio_output
        resolved_channels = self.resolve_audio_output_channel_count()
        return AudioOutputRuntimeConfig(
            output_device=self._runtime_output_device(audio.output_device),
            sample_rate=audio.sample_rate,
            channels=resolved_channels,
            master_output_bus=audio.master_output_bus,
            stream_latency=(
                None
                if audio.latency_profile is AudioLatencyProfile.AUTO
                else audio.latency_profile.value
            ),
            stream_blocksize=audio.blocksize,
            prime_output_buffers_using_stream_callback=audio.prime_output_buffers_using_stream_callback,
        )

    def resolve_audio_output_channel_count(self) -> int | None:
        """Resolve the channel count implied by saved audio preferences."""

        audio = self._preferences.audio_output
        device_channels = self._selected_device_output_channels(audio.output_device)
        if device_channels is not None:
            return int(device_channels)
        if audio.output_channels is not None:
            return int(audio.output_channels)
        return None

    def resolve_ma3_osc_runtime_config(
        self,
        *,
        launch_overrides: AppSettingsLaunchOverrides | None = None,
    ) -> MA3OscRuntimeConfig:
        """Resolve saved and launch-override MA3 OSC settings into one runtime config."""

        overrides = launch_overrides or AppSettingsLaunchOverrides()
        osc = self._preferences.ma3_osc

        receive_enabled = osc.receive.enabled or overrides.ma3_osc_listen_port is not None
        send_enabled = osc.send.enabled or overrides.ma3_osc_command_port is not None
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

    def recent_project_paths(self) -> tuple[Path, ...]:
        """Return the current machine-local recent project list in display order."""

        return tuple(
            Path(path_text)
            for path_text in self._preferences.recent_project_paths
            if str(path_text).strip()
        )

    def remember_recent_project_path(
        self,
        path: str | Path,
        *,
        limit: int = 10,
    ) -> tuple[Path, ...]:
        """Put one project path at the top of the machine-local recent-project list."""

        normalized = self._normalize_recent_project_path(path)
        if normalized is None:
            return self.recent_project_paths()

        max_entries = max(1, int(limit))
        normalized_key = self._recent_project_path_key(normalized)
        ordered: list[str] = [normalized]
        for candidate in self._preferences.recent_project_paths:
            if self._recent_project_path_key(candidate) == normalized_key:
                continue
            text = str(candidate).strip()
            if text:
                ordered.append(text)
            if len(ordered) >= max_entries:
                break

        self.replace_preferences(
            AppPreferences(
                audio_output=self._preferences.audio_output,
                ma3_osc=self._preferences.ma3_osc,
                song_import=self._preferences.song_import,
                pipeline_defaults_by_template=self._preferences.pipeline_defaults_by_template,
                pipeline_profiles_by_template=self._preferences.pipeline_profiles_by_template,
                recent_project_paths=tuple(ordered),
            )
        )
        return self.recent_project_paths()

    def forget_recent_project_path(self, path: str | Path) -> tuple[Path, ...]:
        """Remove one project path from the machine-local recent-project list."""

        normalized = self._normalize_recent_project_path(path)
        if normalized is None:
            return self.recent_project_paths()

        normalized_key = self._recent_project_path_key(normalized)
        filtered = tuple(
            candidate
            for candidate in self._preferences.recent_project_paths
            if self._recent_project_path_key(candidate) != normalized_key
        )
        self.replace_preferences(
            AppPreferences(
                audio_output=self._preferences.audio_output,
                ma3_osc=self._preferences.ma3_osc,
                song_import=self._preferences.song_import,
                pipeline_defaults_by_template=self._preferences.pipeline_defaults_by_template,
                pipeline_profiles_by_template=self._preferences.pipeline_profiles_by_template,
                recent_project_paths=filtered,
            )
        )
        return self.recent_project_paths()

    def _updated_audio_preferences(
        self,
        current: AudioOutputPreferences,
        updates: Mapping[str, object],
    ) -> AudioOutputPreferences:
        output_device = self._device_value(
            updates.get("audio.output_device"),
            current.output_device,
        )
        output_channels = self._optional_positive_int(
            updates.get("audio.output_channels"),
            current.output_channels,
        )
        return AudioOutputPreferences(
            output_device=output_device,
            sample_rate=self._optional_positive_int(
                updates.get("audio.sample_rate"),
                current.sample_rate,
            ),
            output_channels=output_channels,
            master_output_bus=self._output_bus_value(
                updates.get("audio.master_output_bus"),
                current.master_output_bus,
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

    @staticmethod
    def _updated_song_import_preferences(
        current: SongImportPreferences,
        updates: Mapping[str, object],
    ) -> SongImportPreferences:
        pipeline_action_ids = AppSettingsService._updated_song_import_pipeline_action_ids(
            current,
            updates,
        )
        return SongImportPreferences(
            strip_ltc_timecode=bool(
                updates.get("import.strip_ltc_timecode", current.strip_ltc_timecode)
            ),
            name_mode=AppSettingsService._song_import_name_mode(
                updates.get("import.name_mode", current.name_mode)
            ),
            pipeline_action_ids=pipeline_action_ids,
        )

    @staticmethod
    def _song_import_name_mode(value: object) -> SongImportNameMode:
        try:
            raw_value = getattr(value, "value", value) or SongImportNameMode.FILENAME.value
            return SongImportNameMode(str(raw_value).strip())
        except ValueError:
            return SongImportNameMode.FILENAME

    @staticmethod
    def _updated_song_import_pipeline_action_ids(
        current: SongImportPreferences,
        updates: Mapping[str, object],
    ) -> tuple[str, ...]:
        if "import.pipeline_action_ids" in updates:
            raw_action_ids = updates.get("import.pipeline_action_ids")
            action_ids = AppSettingsService._coerce_pipeline_action_ids(raw_action_ids)
        else:
            action_ids = current.pipeline_action_ids

        selected_action_ids = list(canonical_import_pipeline_action_ids(action_ids))
        for descriptor in import_safe_pipeline_action_descriptors():
            key = f"import.pipeline_action.{descriptor.action_id}"
            if key not in updates:
                continue
            selected_action_ids = AppSettingsService._set_action_enabled(
                selected_action_ids,
                descriptor.action_id,
                bool(updates.get(key)),
            )
        return canonical_import_pipeline_action_ids(selected_action_ids)

    @staticmethod
    def _coerce_pipeline_action_ids(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
            return tuple(token for token in tokens if token)
        if isinstance(value, (list, tuple, set)):
            resolved: list[str] = []
            for token in value:
                text = str(token).strip()
                if text:
                    resolved.append(text)
            return tuple(resolved)
        return ()

    @staticmethod
    def _set_action_enabled(
        action_ids: list[str],
        action_id: str,
        enabled: bool,
    ) -> list[str]:
        resolved = [candidate for candidate in action_ids if candidate != action_id]
        if enabled:
            resolved.append(action_id)
        return resolved

    def _validate(self, preferences: AppPreferences) -> None:
        audio = preferences.audio_output
        if audio.sample_rate is not None and audio.sample_rate <= 0:
            raise AppSettingsValidationError("Audio sample rate override must be greater than 0.")
        if audio.output_channels is not None and (
            audio.output_channels < 1 or audio.output_channels > 16
        ):
            raise AppSettingsValidationError(
                "Audio output channels must be between 1 and 16, or Auto."
            )
        parsed_master_buses = parse_output_bus_spans(audio.master_output_bus, reject_invalid=True)
        if not parsed_master_buses or any(
            end_channel > 16 for _start_channel, end_channel in parsed_master_buses
        ):
            raise AppSettingsValidationError(
                "Master output buses must be valid outputs_X_Y routes within outputs 1-16."
            )
        if audio.blocksize is not None and audio.blocksize <= 0:
            raise AppSettingsValidationError("Audio blocksize override must be greater than 0.")

        receive = preferences.ma3_osc.receive
        if receive.enabled and not receive.host.strip():
            raise AppSettingsValidationError(
                "OSC receive bind address is required when receive is enabled."
            )
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
        values: dict[str, object] = {
            "audio.output_device": preferences.audio_output.output_device or "",
            "audio.sample_rate": preferences.audio_output.sample_rate or 0,
            "audio.output_channels": preferences.audio_output.output_channels or 0,
            "audio.master_output_bus": preferences.audio_output.master_output_bus,
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
            "import.name_mode": preferences.song_import.name_mode.value,
            "import.strip_ltc_timecode": preferences.song_import.strip_ltc_timecode,
            "import.pipeline_action_ids": preferences.song_import.pipeline_action_ids,
        }
        configured_action_ids = set(preferences.song_import.pipeline_action_ids)
        for descriptor in import_safe_pipeline_action_descriptors():
            values[f"import.pipeline_action.{descriptor.action_id}"] = (
                descriptor.action_id in configured_action_ids
            )
        return values

    @staticmethod
    def _runtime_output_device(value: str | None) -> int | str | None:
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        return text

    def _selected_device_output_channels(self, output_device: str | None) -> int | None:
        selected_device = str(output_device or "").strip()
        try:
            device_options = self._audio_device_options_provider()
        except Exception:
            device_options = ()
        for option in device_options:
            if str(option.value) != selected_device:
                continue
            try:
                max_outputs = int(option.metadata.get("max_output_channels", 0) or 0)
            except Exception:
                max_outputs = 0
            if max_outputs > 0:
                return max_outputs
        return None

    @staticmethod
    def _output_bus_value(
        value: object,
        current: str,
    ) -> str:
        source = value if value is not None else current
        parsed = parse_output_bus_spans(source, reject_invalid=True)
        if not parsed:
            return DEFAULT_MASTER_OUTPUT_BUS
        if any(end_channel > 16 for _start_channel, end_channel in parsed):
            return current
        tokens = canonical_master_output_buses(
            source,
            default=DEFAULT_MASTER_OUTPUT_BUS,
            reject_invalid=False,
        )
        if not tokens:
            return DEFAULT_MASTER_OUTPUT_BUS
        return ",".join(tokens)

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

    @staticmethod
    def _normalize_recent_project_path(path: str | Path) -> str | None:
        text = str(path).strip()
        return text or None

    @staticmethod
    def _recent_project_path_key(path: str) -> str:
        return str(path).strip().replace("\\", "/").lower()


def build_default_app_settings_service(path: Path | None = None) -> AppSettingsService:
    """Build the canonical app-settings service backed by the local JSON store."""

    from echozero.infrastructure.settings.json_store import JsonAppSettingsStore

    return AppSettingsService(JsonAppSettingsStore(path=path))
