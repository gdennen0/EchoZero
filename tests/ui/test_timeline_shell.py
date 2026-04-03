from echozero.application.timeline.intents import Pause, Play, Seek, ToggleTakeSelector
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import build_variant_presentations


def test_demo_variants_include_take_lanes_open_and_zoom_states():
    variants = build_variant_presentations()
    assert 'take_lanes_open' in variants
    assert 'zoomed_in' in variants
    assert 'zoomed_out' in variants


def test_play_pause_seek_intents_update_presentation():
    demo = build_demo_app()

    stopped = demo.dispatch(Pause())
    assert stopped.is_playing is False

    moved = demo.dispatch(Seek(4.25))
    assert moved.playhead == 4.25

    playing = demo.dispatch(Play())
    assert playing.is_playing is True


def test_realistic_fixture_contains_song_stems_and_drum_classifiers():
    demo = build_demo_app()
    presentation = demo.presentation()
    titles = {layer.title for layer in presentation.layers}

    assert {'Song', 'Drums', 'Bass', 'Vocals', 'Other', 'Kick', 'Snare', 'HiHat', 'Clap'} <= titles


def test_take_lanes_exist_without_inline_action_requirements():
    demo = build_demo_app()
    presentation = demo.presentation()
    drums = next(layer for layer in presentation.layers if layer.title == 'Drums')
    kick = next(layer for layer in presentation.layers if layer.title == 'Kick')

    assert len(drums.takes) >= 1
    assert drums.takes[0].kind.name == 'AUDIO'
    assert len(kick.takes) >= 1
    assert kick.takes[0].kind.name == 'EVENT'


def test_toggle_take_selector_round_trips():
    demo = build_demo_app()
    song = next(layer for layer in demo.presentation().layers if layer.title == 'Song')

    expanded = demo.dispatch(ToggleTakeSelector(song.layer_id))
    expanded_song = next(layer for layer in expanded.layers if layer.title == 'Song')
    assert expanded_song.is_expanded is True

    collapsed = demo.dispatch(ToggleTakeSelector(song.layer_id))
    collapsed_song = next(layer for layer in collapsed.layers if layer.title == 'Song')
    assert collapsed_song.is_expanded is False


def test_fixture_has_muted_and_soloed_layers_for_daw_state_rendering():
    demo = build_demo_app()
    presentation = demo.presentation()
    assert any(layer.muted for layer in presentation.layers)
    assert any(layer.soloed for layer in presentation.layers)
