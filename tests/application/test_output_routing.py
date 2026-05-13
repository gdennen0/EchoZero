"""Shared output-routing vocabulary coverage."""

from echozero.output_routing import (
    canonical_layer_output_bus,
    canonical_master_output_buses,
    output_bus_channel_spans,
    output_bus_label,
    output_bus_options,
    parse_output_bus_spans,
)


def test_master_output_buses_canonicalize_comma_values_and_dedupe() -> None:
    assert canonical_master_output_buses(
        " outputs_2_2,outputs_1_1,outputs_2_2 "
    ) == ("outputs_2_2", "outputs_1_1")


def test_layer_output_bus_is_single_canonical_route() -> None:
    assert canonical_layer_output_bus(" Outputs_3_4 ") == "outputs_3_4"
    assert canonical_layer_output_bus("outputs_3_4,outputs_1_1") == "outputs_3_4"


def test_layer_output_bus_bounds_to_device_channels() -> None:
    assert (
        canonical_layer_output_bus(
            "outputs_3_4",
            max_channel=3,
            clamp_to_channels=True,
            reject_invalid=True,
        )
        == "outputs_3_3"
    )
    assert canonical_layer_output_bus("outputs_7_8", max_channel=4) is None


def test_output_bus_options_and_labels_share_route_vocabulary() -> None:
    assert [(route.token, route.label) for route in output_bus_options(3)] == [
        ("outputs_1_1", "Output 1"),
        ("outputs_2_2", "Output 2"),
        ("outputs_3_3", "Output 3"),
    ]
    assert output_bus_label("outputs_3_4") == "Outputs 3/4"
    assert output_bus_label("outputs_1_4") == "Outputs 1-4"


def test_parse_output_bus_spans_rejects_invalid_when_requested() -> None:
    assert parse_output_bus_spans("outputs_1_1,invalid", reject_invalid=True) == ()
    assert parse_output_bus_spans("outputs_1_1,invalid") == ((1, 1),)


def test_output_bus_channel_spans_mirror_master_defaults_only() -> None:
    assert output_bus_channel_spans(
        None,
        4,
        default_output_buses="outputs_1_1,outputs_3_3",
    ) == ((0, 1), (2, 1))
    assert output_bus_channel_spans("outputs_1_1,outputs_3_3", 4) == ((0, 1),)
