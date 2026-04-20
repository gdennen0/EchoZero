"""
OBS demo capture control for EchoZero recordings.
Exists because demo recording should be repeatable after the app is open, not a one-off manual OBS setup.
Uses the local OBS websocket to bind the EZ window source and start/stop recording on demand.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import obsws_python as obs


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4455
DEFAULT_PASSWORD = "ez_obs_local_2026"
DEFAULT_SCENE = "EchoZero Capture"
DEFAULT_WINDOW_SOURCE = "EchoZero Window"
DEFAULT_DISPLAY_SOURCE = "EchoZero Display"
DEFAULT_AUDIO_SOURCE = "System Audio SCK"
DEFAULT_MIC_SOURCE = "Mic/Aux"


@dataclass(frozen=True, slots=True)
class WindowCandidate:
    """One OBS-visible window target that can be used for window capture."""

    name: str
    value: int
    enabled: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for reusable OBS demo capture control."""
    parser = argparse.ArgumentParser(description="Control OBS demo capture for EchoZero.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--window-source", default=DEFAULT_WINDOW_SOURCE)
    parser.add_argument("--display-source", default=DEFAULT_DISPLAY_SOURCE)
    parser.add_argument("--audio-source", default=DEFAULT_AUDIO_SOURCE)
    parser.add_argument("--mic-source", default=DEFAULT_MIC_SOURCE)

    sub = parser.add_subparsers(dest="command", required=True)

    list_windows = sub.add_parser("list-windows", help="List OBS-visible window capture candidates.")
    list_windows.add_argument("--filter", default="", help="Case-insensitive substring filter for window names.")

    bind = sub.add_parser("bind-window", help="Bind the OBS window source to a live app window.")
    bind.add_argument("--match", required=True, help="Case-insensitive substring to match against OBS window names.")
    bind.add_argument("--show-cursor", action="store_true", help="Show the mouse cursor in the capture.")

    bind_display = sub.add_parser("bind-display", help="Use full-display capture for modal dialogs and sheets.")
    bind_display.add_argument("--show-cursor", action="store_true", help="Show the mouse cursor in the capture.")

    sub.add_parser("start", help="Start OBS recording.")
    sub.add_parser("stop", help="Stop OBS recording.")
    sub.add_parser("status", help="Show current OBS capture state.")
    return parser


def create_client(args: argparse.Namespace) -> obs.ReqClient:
    """Create a websocket client for the locally configured OBS instance."""
    return obs.ReqClient(
        host=args.host,
        port=args.port,
        password=args.password,
        timeout=5,
    )


def ensure_scene_and_inputs(
    client: obs.ReqClient,
    *,
    scene_name: str,
    window_source_name: str,
    display_source_name: str,
    audio_source_name: str,
) -> None:
    """Ensure the expected demo scene and capture inputs exist in OBS."""
    scenes = client.get_scene_list()
    scene_names = {_scene_name(item) for item in scenes.scenes}
    if scene_name not in scene_names:
        client.create_scene(scene_name)
    client.set_current_program_scene(scene_name)

    inputs = client.get_input_list().inputs
    input_names = {item["inputName"] for item in inputs}
    if window_source_name not in input_names:
        client.create_input(scene_name, window_source_name, "window_capture", {"show_cursor": True}, True)
    if display_source_name not in input_names:
        client.create_input(scene_name, display_source_name, "display_capture", {"show_cursor": True}, True)
    if audio_source_name not in input_names:
        input_kinds = set(client.get_input_kind_list(unversioned=False).input_kinds)
        audio_kind = "sck_audio_capture" if "sck_audio_capture" in input_kinds else "coreaudio_output_capture"
        client.create_input(scene_name, audio_source_name, audio_kind, {}, True)


def _scene_item_id(client: obs.ReqClient, *, scene_name: str, source_name: str) -> int:
    items = client.get_scene_item_list(scene_name).scene_items
    for item in items:
        if item["sourceName"] == source_name:
            return int(item["sceneItemId"])
    raise RuntimeError(f"Scene item not found for source '{source_name}' in scene '{scene_name}'.")


def fit_source_to_canvas(
    client: obs.ReqClient,
    *,
    scene_name: str,
    source_name: str,
) -> dict[str, object]:
    """Scale the source to fit fully inside the OBS canvas while preserving aspect ratio."""
    video = client.get_video_settings()
    canvas_width = float(video.base_width)
    canvas_height = float(video.base_height)
    item_id = _scene_item_id(client, scene_name=scene_name, source_name=source_name)
    item = next(
        item
        for item in client.get_scene_item_list(scene_name).scene_items
        if item["sceneItemId"] == item_id
    )
    transform = item["sceneItemTransform"]
    source_width = float(transform["sourceWidth"] or transform["width"] or canvas_width)
    source_height = float(transform["sourceHeight"] or transform["height"] or canvas_height)
    scale = min(canvas_width / source_width, canvas_height / source_height)
    scaled_width = source_width * scale
    scaled_height = source_height * scale
    new_transform = {
        "positionX": (canvas_width - scaled_width) / 2.0,
        "positionY": (canvas_height - scaled_height) / 2.0,
        "scaleX": scale,
        "scaleY": scale,
        "rotation": 0.0,
        "alignment": 5,
        "boundsType": "OBS_BOUNDS_NONE",
        "cropLeft": 0,
        "cropTop": 0,
        "cropRight": 0,
        "cropBottom": 0,
        "cropToBounds": False,
    }
    client.set_scene_item_transform(scene_name, item_id, new_transform)
    return {
        "scene": scene_name,
        "source": source_name,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "source_width": source_width,
        "source_height": source_height,
        "scale": scale,
    }


def set_scene_item_enabled(
    client: obs.ReqClient,
    *,
    scene_name: str,
    source_name: str,
    enabled: bool,
) -> None:
    """Show or hide one scene item by source name."""
    item_id = _scene_item_id(client, scene_name=scene_name, source_name=source_name)
    client.set_scene_item_enabled(scene_name, item_id, enabled)


def mute_input_if_present(client: obs.ReqClient, *, input_name: str) -> bool:
    """Mute one OBS input if it exists."""
    input_names = {item["inputName"] for item in client.get_input_list().inputs}
    if input_name not in input_names:
        return False
    client.set_input_mute(input_name, True)
    return True


def list_window_candidates(client: obs.ReqClient, *, source_name: str) -> list[WindowCandidate]:
    """Return live window-capture candidates exposed by OBS for the given source."""
    response = client.get_input_properties_list_property_items(source_name, "window")
    candidates: list[WindowCandidate] = []
    for item in response.property_items:
        candidates.append(
            WindowCandidate(
                name=str(item["itemName"]),
                value=int(item["itemValue"]),
                enabled=bool(item["itemEnabled"]),
            )
        )
    return candidates


def choose_window(candidates: Iterable[WindowCandidate], *, match: str) -> WindowCandidate:
    """Pick one window candidate by case-insensitive substring match."""
    normalized_match = match.strip().lower()
    if not normalized_match:
        raise ValueError("bind-window requires a non-empty --match value.")

    matches = [candidate for candidate in candidates if normalized_match in candidate.name.lower() and candidate.enabled]
    if not matches:
        raise RuntimeError(f"No OBS window candidate matched '{match}'.")
    if len(matches) > 1:
        names = ", ".join(candidate.name for candidate in matches[:10])
        raise RuntimeError(f"Multiple OBS windows matched '{match}'. Narrow the pattern: {names}")
    return matches[0]


def bind_window(
    client: obs.ReqClient,
    *,
    scene_name: str,
    source_name: str,
    target: WindowCandidate,
    show_cursor: bool,
    mic_source_name: str,
) -> dict[str, object]:
    """Point the OBS window capture source at one selected live window."""
    settings = {
        "window": target.value,
        "show_cursor": show_cursor,
    }
    client.set_input_settings(source_name, settings, True)
    fit = fit_source_to_canvas(client, scene_name=scene_name, source_name=source_name)
    set_scene_item_enabled(client, scene_name=scene_name, source_name=source_name, enabled=True)
    mic_muted = mute_input_if_present(client, input_name=mic_source_name)
    return {
        "source": source_name,
        "window_name": target.name,
        "window_id": target.value,
        "show_cursor": show_cursor,
        "fit": fit,
        "mic_muted": mic_muted,
    }


def bind_display(
    client: obs.ReqClient,
    *,
    scene_name: str,
    source_name: str,
    window_source_name: str,
    show_cursor: bool,
    mic_source_name: str,
) -> dict[str, object]:
    """Use full display capture to include app-modal dialogs and sheets."""
    client.set_input_settings(source_name, {"show_cursor": show_cursor}, True)
    fit = fit_source_to_canvas(client, scene_name=scene_name, source_name=source_name)
    set_scene_item_enabled(client, scene_name=scene_name, source_name=source_name, enabled=True)
    set_scene_item_enabled(client, scene_name=scene_name, source_name=window_source_name, enabled=False)
    mic_muted = mute_input_if_present(client, input_name=mic_source_name)
    return {
        "source": source_name,
        "show_cursor": show_cursor,
        "fit": fit,
        "window_source_disabled": window_source_name,
        "mic_muted": mic_muted,
    }


def build_status(client: obs.ReqClient, *, scene_name: str) -> dict[str, object]:
    """Collect the current demo capture status from OBS."""
    version = client.get_version()
    scenes = client.get_scene_list()
    record = client.get_record_status()
    return {
        "obs_version": version.obs_version,
        "rpc_version": version.rpc_version,
        "current_scene": scenes.current_program_scene_name,
        "record_active": record.output_active,
        "record_paused": record.output_paused,
        "scene_expected": scene_name,
    }


def _scene_name(item: dict[str, object] | object) -> str:
    if isinstance(item, dict):
        return str(item["sceneName"])
    return str(getattr(item, "sceneName"))


def main(argv: list[str] | None = None) -> int:
    """Run the OBS demo capture CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    client = create_client(args)

    ensure_scene_and_inputs(
        client,
        scene_name=args.scene,
        window_source_name=args.window_source,
        display_source_name=args.display_source,
        audio_source_name=args.audio_source,
    )

    if args.command == "list-windows":
        candidates = list_window_candidates(client, source_name=args.window_source)
        filtered = [
            candidate
            for candidate in candidates
            if args.filter.strip().lower() in candidate.name.lower()
        ]
        print(json.dumps([asdict(candidate) for candidate in filtered], indent=2))
        return 0

    if args.command == "bind-window":
        candidates = list_window_candidates(client, source_name=args.window_source)
        match = choose_window(candidates, match=args.match)
        result = bind_window(
            client,
            scene_name=args.scene,
            source_name=args.window_source,
            target=match,
            show_cursor=bool(args.show_cursor),
            mic_source_name=args.mic_source,
        )
        set_scene_item_enabled(client, scene_name=args.scene, source_name=args.display_source, enabled=False)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "bind-display":
        result = bind_display(
            client,
            scene_name=args.scene,
            source_name=args.display_source,
            window_source_name=args.window_source,
            show_cursor=bool(args.show_cursor),
            mic_source_name=args.mic_source,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "start":
        client.start_record()
        print(json.dumps(build_status(client, scene_name=args.scene), indent=2))
        return 0

    if args.command == "stop":
        client.stop_record()
        print(json.dumps(build_status(client, scene_name=args.scene), indent=2))
        return 0

    if args.command == "status":
        print(json.dumps(build_status(client, scene_name=args.scene), indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
