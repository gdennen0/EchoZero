from echozero.ui.qt.timeline.fixture_loader import fixture_path, load_realistic_timeline_fixture
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, fixture_color, fixture_take_action_label


def test_realistic_fixture_file_exists():
    assert fixture_path().exists()


def test_realistic_fixture_contains_core_layers_and_takes():
    presentation = load_realistic_timeline_fixture()
    by_title = {layer.title: layer for layer in presentation.layers}

    assert {"Song", "Drums", "Bass", "Vocals", "Other", "Kick", "Snare", "HiHat", "Clap"} <= set(by_title)
    assert by_title["Drums"].kind.name == "AUDIO"
    assert len(by_title["Drums"].takes) >= 2
    assert by_title["Kick"].kind.name == "EVENT"
    assert len(by_title["Kick"].takes) >= 2


def test_realistic_fixture_take_actions_include_overwrite_and_merge():
    presentation = load_realistic_timeline_fixture()

    takes_with_actions = [take for layer in presentation.layers for take in layer.takes if take.actions]
    assert takes_with_actions

    for take in takes_with_actions:
        action_ids = {action.action_id for action in take.actions}
        assert {"overwrite_main", "merge_main"} <= action_ids
        assert {action.label for action in take.actions} <= set(TIMELINE_STYLE.fixture.take_action_labels.values())


def test_realistic_fixture_event_ids_are_unique():
    presentation = load_realistic_timeline_fixture()

    seen: set[str] = set()
    for layer in presentation.layers:
        for event in layer.events:
            assert str(event.event_id) not in seen
            seen.add(str(event.event_id))
        for take in layer.takes:
            for event in take.events:
                assert str(event.event_id) not in seen
                seen.add(str(event.event_id))


def test_realistic_fixture_has_stale_manual_and_sync_signals():
    presentation = load_realistic_timeline_fixture()

    assert any(layer.status.stale for layer in presentation.layers)
    assert any(layer.status.manually_modified for layer in presentation.layers)
    assert any("sync" in layer.title.lower() or layer.status.sync_label for layer in presentation.layers)


def test_realistic_fixture_resolves_layer_and_event_colors_from_style_tokens():
    presentation = load_realistic_timeline_fixture()
    by_title = {layer.title: layer for layer in presentation.layers}

    assert by_title["Song"].color == fixture_color("song")
    assert by_title["Drums"].color == fixture_color("drums")
    assert by_title["Kick"].color == fixture_color("kick")
    assert by_title["Clap"].color == fixture_color("clap")
    assert by_title["MA3 Sync"].color == fixture_color("sync")

    kick_event = by_title["Kick"].events[0]
    clap_take_event = by_title["Clap"].takes[0].events[0]
    assert kick_event.color == by_title["Kick"].color
    assert clap_take_event.color == by_title["Clap"].color


def test_realistic_fixture_missing_action_labels_are_backfilled_from_style_tokens():
    presentation = load_realistic_timeline_fixture()
    take = next(layer.takes[0] for layer in presentation.layers if layer.takes)

    labels_by_id = {action.action_id: action.label for action in take.actions}
    assert labels_by_id["overwrite_main"] == fixture_take_action_label("overwrite_main")
    assert labels_by_id["merge_main"] == fixture_take_action_label("merge_main")
