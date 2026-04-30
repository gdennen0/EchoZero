import json
from dataclasses import replace

from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_runtime_state import (
    load_project_runtime_state,
    persist_project_runtime_state,
)
from echozero.ui.qt.timeline.demo_app import build_demo_app


def test_runtime_state_persists_layer_header_width_v2(tmp_path):
    storage = ProjectStorage.create_new(name="Runtime State", working_dir_root=tmp_path)
    try:
        presentation = replace(
            build_demo_app().presentation(),
            active_song_id="song_1",
            active_song_version_id="version_1",
        )
        persist_project_runtime_state(
            storage,
            presentation=presentation,
            playhead=12.5,
            layer_header_width_px=444,
        )

        loaded = load_project_runtime_state(storage)
        assert loaded.playhead == 12.5
        assert loaded.layer_header_width_px == 444

        runtime_state_path = storage.working_dir / "app_shell_runtime_state.json"
        payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "echozero.app_shell_runtime_state.v1"
        assert payload["state"]["layer_header_width_px"] == 444
    finally:
        storage.close()


def test_runtime_state_load_falls_back_to_v1_metadata_key(tmp_path):
    storage = ProjectStorage.create_new(name="Runtime State V1", working_dir_root=tmp_path)
    try:
        payload = {
            "active_song_id": "song_legacy",
            "active_song_version_id": "version_legacy",
            "playhead": 4.0,
            "pixels_per_second": 140.0,
            "scroll_x": 220.0,
            "scroll_y": 0.0,
            "layer_header_width_px": 388,
        }
        with storage.locked():
            storage.db.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                ("app_shell_runtime_state.v1", json.dumps(payload)),
            )

        loaded = load_project_runtime_state(storage)
        assert str(loaded.active_song_id) == "song_legacy"
        assert str(loaded.active_song_version_id) == "version_legacy"
        assert loaded.layer_header_width_px == 388
    finally:
        storage.close()


def test_runtime_state_load_prefers_json_file_over_metadata(tmp_path):
    storage = ProjectStorage.create_new(name="Runtime State JSON Priority", working_dir_root=tmp_path)
    try:
        with storage.locked():
            storage.db.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                (
                    "app_shell_runtime_state.v2",
                    json.dumps(
                        {
                            "active_song_id": "song_meta",
                            "active_song_version_id": "version_meta",
                            "playhead": 9.0,
                            "layer_header_width_px": 390,
                        }
                    ),
                ),
            )
        (storage.working_dir / "app_shell_runtime_state.json").write_text(
            json.dumps(
                {
                    "schema": "echozero.app_shell_runtime_state.v1",
                    "state": {
                        "active_song_id": "song_json",
                        "active_song_version_id": "version_json",
                        "playhead": 3.5,
                        "pixels_per_second": 123.0,
                        "scroll_x": 12.0,
                        "scroll_y": 0.0,
                        "layer_header_width_px": 410,
                    },
                }
            ),
            encoding="utf-8",
        )

        loaded = load_project_runtime_state(storage)
        assert str(loaded.active_song_id) == "song_json"
        assert str(loaded.active_song_version_id) == "version_json"
        assert loaded.playhead == 3.5
        assert loaded.layer_header_width_px == 410
    finally:
        storage.close()
