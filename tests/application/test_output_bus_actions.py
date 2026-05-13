"""Inspector layer-routing action coverage."""

from echozero.application.presentation.inspector_contract_context_actions import (
    _available_output_bus_tokens,
    _layer_routing_settings_action,
)
from echozero.application.presentation.models import LayerPresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId


def _audio_layer(*, output_bus: str | None = None) -> LayerPresentation:
    return LayerPresentation(
        layer_id=LayerId("layer_song"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
        output_bus=output_bus,
    )


def test_available_output_bus_tokens_are_single_physical_outputs() -> None:
    assert _available_output_bus_tokens(4) == (
        "outputs_1_1",
        "outputs_2_2",
        "outputs_3_3",
        "outputs_4_4",
    )


def test_layer_routing_settings_action_uses_single_entrypoint() -> None:
    layer = _audio_layer(output_bus="outputs_7_8")
    action = _layer_routing_settings_action(layer)

    assert action.action_id == "layer.routing_settings"
    assert action.label == "Layer Routing Settings"
    assert action.params == {"layer_id": LayerId("layer_song")}
