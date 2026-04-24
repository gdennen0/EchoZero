"""App settings page builders for machine-local EchoZero preferences.
Exists because settings page structure should stay reusable and separate from persistence rules.
Connects typed app preferences to neutral settings-page contracts for Qt and future surfaces.
"""

from __future__ import annotations

from collections.abc import Callable

from echozero.application.settings.contracts import (
    SettingsField,
    SettingsFieldSurface,
    SettingsFieldWidget,
    SettingsOption,
    SettingsPage,
    SettingsSection,
)
from echozero.application.settings.models import (
    AppPreferences,
    AudioLatencyProfile,
    AudioOutputPreferences,
    MA3OscPreferences,
    OscReceivePreferences,
    OscSendPreferences,
)


def list_audio_output_device_options() -> tuple[SettingsOption, ...]:
    """Return available sounddevice output choices for the current machine."""

    options = [SettingsOption(value="", label="System Default")]
    try:
        import sounddevice as sd

        default_device = sd.default.device[1]
        for index, raw_info in enumerate(sd.query_devices()):
            device = dict(raw_info)
            if int(device.get("max_output_channels", 0)) <= 0:
                continue
            label = str(device.get("name") or f"Output Device {index}")
            if index == default_device:
                label = f"{label} (System Default)"
            options.append(SettingsOption(value=str(index), label=label))
    except Exception:
        pass
    return tuple(options)


def build_app_settings_page(
    preferences: AppPreferences,
    *,
    audio_device_options_provider: Callable[[], tuple[SettingsOption, ...]] = (
        list_audio_output_device_options
    ),
    include_hidden: bool = False,
) -> SettingsPage:
    """Render app preferences into one reusable neutral settings page."""

    return SettingsPage(
        key="app_preferences",
        title="Application Settings",
        summary=(
            "Machine-local defaults for EchoZero audio output and OSC integration. "
            "This surface edits the local config file only."
        ),
        sections=tuple(
            section
            for section in (
                _audio_section(
                    preferences.audio_output,
                    audio_device_options_provider=audio_device_options_provider,
                    include_hidden=include_hidden,
                ),
                _osc_receive_section(
                    preferences.ma3_osc.receive,
                    include_hidden=include_hidden,
                ),
                _osc_send_section(
                    preferences.ma3_osc.send,
                    include_hidden=include_hidden,
                ),
            )
            if section.fields
        ),
        warnings=(
            "Changes are saved to the local config JSON only.",
            "Restart EchoZero to use updated audio or OSC settings.",
        ),
    )


def _audio_section(
    audio: AudioOutputPreferences,
    *,
    audio_device_options_provider: Callable[[], tuple[SettingsOption, ...]],
    include_hidden: bool,
) -> SettingsSection:
    selected_device = audio.output_device or ""
    device_options = list(audio_device_options_provider())
    if selected_device and all(str(option.value) != selected_device for option in device_options):
        device_options.append(
            SettingsOption(value=selected_device, label=f"Saved Device ({selected_device})")
        )

    fields = (
        SettingsField(
            key="audio.output_device",
            label="Output Device",
            value=selected_device,
            default_value="",
            widget=SettingsFieldWidget.DROPDOWN,
            description="Choose a specific output device or keep the system default.",
            options=tuple(device_options),
        ),
        SettingsField(
            key="audio.latency_profile",
            label="Latency Profile",
            value=audio.latency_profile.value,
            default_value=AudioLatencyProfile.AUTO.value,
            widget=SettingsFieldWidget.DROPDOWN,
            description="Auto follows EchoZero defaults. Low reduces latency; high favors stability.",
            options=(
                SettingsOption(value=AudioLatencyProfile.AUTO.value, label="Auto"),
                SettingsOption(value=AudioLatencyProfile.LOW.value, label="Low"),
                SettingsOption(value=AudioLatencyProfile.HIGH.value, label="High"),
            ),
        ),
        SettingsField(
            key="audio.sample_rate",
            label="Sample Rate Override",
            value=audio.sample_rate or 0,
            default_value=0,
            widget=SettingsFieldWidget.NUMBER,
            description="0 keeps the selected device's preferred sample rate.",
            units="Hz",
            min_value=0,
            max_value=384000,
            step=1000,
            surface=SettingsFieldSurface.ADVANCED,
        ),
        SettingsField(
            key="audio.output_channels",
            label="Output Channels",
            value=audio.output_channels or 0,
            default_value=0,
            widget=SettingsFieldWidget.DROPDOWN,
            description="Auto follows the device. EchoZero currently supports mono or stereo output.",
            options=(
                SettingsOption(value=0, label="Auto"),
                SettingsOption(value=1, label="Mono"),
                SettingsOption(value=2, label="Stereo"),
            ),
            surface=SettingsFieldSurface.ADVANCED,
        ),
        SettingsField(
            key="audio.blocksize",
            label="Blocksize Override",
            value=audio.blocksize or 0,
            default_value=0,
            widget=SettingsFieldWidget.NUMBER,
            description="0 lets the host choose the callback size for best compatibility.",
            units="frames",
            min_value=0,
            max_value=32768,
            step=64,
            surface=SettingsFieldSurface.ADVANCED,
        ),
        SettingsField(
            key="audio.prime_output_buffers_using_stream_callback",
            label="Prime Output Buffers In Callback",
            value=audio.prime_output_buffers_using_stream_callback,
            default_value=True,
            widget=SettingsFieldWidget.TOGGLE,
            description="Leave enabled unless debugging device startup behavior.",
            surface=SettingsFieldSurface.ADVANCED,
        ),
    )
    return _section(
        key="audio",
        title="Audio Output",
        description="Saved playback defaults for this machine.",
        fields=fields,
        include_hidden=include_hidden,
    )


def _osc_receive_section(
    receive: OscReceivePreferences,
    *,
    include_hidden: bool,
) -> SettingsSection:
    return _section(
        key="osc_receive",
        title="OSC Receive",
        description="Packet Sender-style receive binding for EchoZero's incoming OSC listener.",
        fields=(
            SettingsField(
                key="osc_receive.enabled",
                label="Receive OSC",
                value=receive.enabled,
                default_value=False,
                widget=SettingsFieldWidget.TOGGLE,
                description="Enable EchoZero's incoming OSC listener.",
            ),
            SettingsField(
                key="osc_receive.host",
                label="Bind Address",
                value=receive.host,
                default_value="127.0.0.1",
                description="IP address or hostname EchoZero should bind for incoming OSC.",
            ),
            SettingsField(
                key="osc_receive.port",
                label="Bind Port",
                value=receive.port,
                default_value=0,
                widget=SettingsFieldWidget.NUMBER,
                description="0 requests an ephemeral receive port on launch.",
                units="port",
                min_value=0,
                max_value=65535,
                step=1,
            ),
        ),
        include_hidden=include_hidden,
    )


def _osc_send_section(
    send: OscSendPreferences,
    *,
    include_hidden: bool,
) -> SettingsSection:
    return _section(
        key="osc_send",
        title="OSC Send",
        description="Packet Sender-style destination defaults for EchoZero's outbound OSC traffic.",
        fields=(
            SettingsField(
                key="osc_send.enabled",
                label="Send OSC",
                value=send.enabled,
                default_value=False,
                widget=SettingsFieldWidget.TOGGLE,
                description="Enable EchoZero's outbound OSC sender.",
            ),
            SettingsField(
                key="osc_send.host",
                label="Target Address",
                value=send.host,
                default_value="127.0.0.1",
                description="IP address or hostname EchoZero should send OSC to.",
            ),
            SettingsField(
                key="osc_send.port",
                label="Target Port",
                value=send.port or 0,
                default_value=0,
                widget=SettingsFieldWidget.NUMBER,
                description="Set the outbound OSC destination port.",
                units="port",
                min_value=0,
                max_value=65535,
                step=1,
            ),
        ),
        include_hidden=include_hidden,
    )


def _section(
    *,
    key: str,
    title: str,
    description: str,
    fields: tuple[SettingsField, ...],
    include_hidden: bool,
) -> SettingsSection:
    resolved_fields = tuple(
        field
        for field in fields
        if include_hidden or field.surface is not SettingsFieldSurface.HIDDEN
    )
    return SettingsSection(
        key=key,
        title=title,
        description=description,
        fields=resolved_fields,
    )
