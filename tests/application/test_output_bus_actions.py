"""Inspector layer-routing action coverage."""

from echozero.application.presentation.inspector_contract_context_actions import (
    _available_output_bus_tokens,
    _layer_routing_settings_action,
)
from echozero.application.presentation.models import LayerPresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId
from echozero.output_routing import (
    canonical_layer_output_bus,
    output_bus_channel_spans,
    output_bus_label,
)


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


def test_layer_output_bus_preserves_master_plus_explicit_routes() -> None:
    assert (
        canonical_layer_output_bus("master,outputs_3_3,outputs_4_4", reject_invalid=True)
        == "master,outputs_3_3,outputs_4_4"
    )
    assert output_bus_label("master,outputs_3_3") == "Master Output, Output 3"
    assert output_bus_channel_spans(
        "master,outputs_3_3",
        4,
        default_output_buses=("outputs_1_2",),
    ) == ((0, 2), (2, 1))


def test_layer_output_bus_none_disables_master_and_explicit_routes() -> None:
    assert canonical_layer_output_bus("none", reject_invalid=True) == "none"
    assert output_bus_label("none") == "No Output"
    assert (
        output_bus_channel_spans(
            "none",
            4,
            default_output_buses=("outputs_1_2",),
        )
        == ()
    )
