from echozero.ui.qt.timeline.fixture_loader import fixture_path, load_realistic_timeline_fixture


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
