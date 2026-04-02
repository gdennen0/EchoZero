from echozero.application.timeline.intents import Pause, Play, Seek, SelectTake, ToggleTakeSelector
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import build_variant_presentations


def test_demo_variants_include_take_menu_open():
    variants = build_variant_presentations()
    assert "take_menu_open" in variants
    assert variants["take_menu_open"].layers[0].is_expanded is True


def test_play_pause_seek_intents_update_presentation():
    demo = build_demo_app()

    stopped = demo.dispatch(Pause())
    assert stopped.is_playing is False

    moved = demo.dispatch(Seek(4.25))
    assert moved.playhead == 4.25

    playing = demo.dispatch(Play())
    assert playing.is_playing is True


def test_select_take_changes_active_take_and_selection():
    demo = build_demo_app()
    layer = demo.timeline.layers[0]
    alt_take = layer.takes[1]

    presentation = demo.dispatch(SelectTake(layer.id, alt_take.id))

    assert layer.active_take_id == alt_take.id
    assert presentation.selected_take_id == alt_take.id
    assert presentation.layers[0].subtitle == alt_take.name


def test_toggle_take_selector_round_trips():
    demo = build_demo_app()
    layer = demo.timeline.layers[0]

    expanded = demo.dispatch(ToggleTakeSelector(layer.id))
    assert expanded.layers[0].is_expanded is True

    collapsed = demo.dispatch(ToggleTakeSelector(layer.id))
    assert collapsed.layers[0].is_expanded is False
