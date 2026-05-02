"""
Playback process service entrypoint for IPC runtime hosting.
Exists because process lifecycle wiring and signal handling should stay outside service core logic.
Connects command-line process bootstrap to PlaybackProcessService startup and shutdown.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from echozero.application.playback.process_service import PlaybackProcessService
from echozero.application.playback.process_shared import PLAYBACK_IPC_HOST
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.errors import InfrastructureError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run process-isolated playback runtime host.")
    parser.add_argument("--host", type=str, default=PLAYBACK_IPC_HOST)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ws-port", type=int, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--audio-config-json", type=Path, default=None)
    return parser.parse_args(argv)


def _load_audio_config(config_path: Path | None) -> AudioOutputRuntimeConfig | None:
    if config_path is None:
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InfrastructureError("Invalid playback audio config payload")
    return AudioOutputRuntimeConfig(
        output_device=payload.get("output_device"),
        sample_rate=(int(payload["sample_rate"]) if payload.get("sample_rate") is not None else None),
        channels=(int(payload["channels"]) if payload.get("channels") is not None else None),
        stream_latency=payload.get("stream_latency"),
        stream_blocksize=(
            int(payload["stream_blocksize"])
            if payload.get("stream_blocksize") is not None
            else None
        ),
        prime_output_buffers_using_stream_callback=bool(
            payload.get("prime_output_buffers_using_stream_callback", True)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    service = PlaybackProcessService(
        host=str(args.host or PLAYBACK_IPC_HOST),
        port=int(args.port),
        ws_port=int(args.ws_port),
        token=str(args.token),
        base_audio_config=_load_audio_config(args.audio_config_json),
    )

    def _request_shutdown(*_unused: object) -> None:
        service.request_shutdown()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    return service.run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
