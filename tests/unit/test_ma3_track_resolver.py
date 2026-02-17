from src.features.show_manager.application.ma3_track_resolver import (
    normalize_track_name,
    find_track_by_name,
)


class DummyTrack:
    def __init__(self, name):
        self.name = name


def test_normalize_track_name():
    assert normalize_track_name("  Kick  Drum ") == "kick drum"
    assert normalize_track_name("Kick_Drum") == "kick drum"
    assert normalize_track_name("Kick-Drum") == "kick drum"
    assert normalize_track_name("") == ""
    assert normalize_track_name(None) == ""


def test_find_track_by_name():
    tracks = [DummyTrack("Kick Drum"), DummyTrack("Snare")]
    assert find_track_by_name(tracks, "kick drum") is tracks[0]
    assert find_track_by_name(tracks, "SNARE") is tracks[1]

    dict_tracks = [{"name": "Hi Hat"}, {"name": "Clap"}]
    assert find_track_by_name(dict_tracks, "hi  hat") == dict_tracks[0]

    assert find_track_by_name(tracks, "Missing") is None
