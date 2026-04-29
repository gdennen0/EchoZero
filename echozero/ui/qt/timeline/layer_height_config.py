"""Timeline layer-height config loader.
Exists to keep lane sizing tunable via JSON without embedding parsing logic in canvas widgets.
Connects FEEL fallback constants to per-layer-kind defaults used by the timeline render path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from echozero.application.shared.enums import LayerKind
from echozero.ui.FEEL import LAYER_ROW_HEIGHT_PX, LAYER_ROW_MIN_HEIGHT_PX, TAKE_ROW_HEIGHT_PX


@dataclass(frozen=True, slots=True)
class LayerHeightConfig:
    default_main_row_height_px: int
    take_row_height_px: int
    min_main_row_height_px: int
    max_main_row_height_px: int
    resize_handle_hit_padding_px: int
    layer_kind_main_row_height_px: Mapping[LayerKind, int]


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "layer_heights.json"
_DEFAULT_LAYER_KIND_HEIGHTS = {
    LayerKind.AUDIO: LAYER_ROW_HEIGHT_PX,
    LayerKind.EVENT: LAYER_ROW_HEIGHT_PX,
    LayerKind.MARKER: LAYER_ROW_HEIGHT_PX,
    LayerKind.SECTION: LAYER_ROW_HEIGHT_PX,
    LayerKind.AUTOMATION: LAYER_ROW_HEIGHT_PX,
    LayerKind.REFERENCE: LAYER_ROW_HEIGHT_PX,
    LayerKind.GROUP: LAYER_ROW_HEIGHT_PX,
    LayerKind.SYNC: LAYER_ROW_HEIGHT_PX,
}


def _coerce_int(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            return int(stripped)
        except ValueError:
            return fallback
    return fallback


def _read_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_layer_height_config(path: Path | None = None) -> LayerHeightConfig:
    source_path = _DEFAULT_CONFIG_PATH if path is None else path
    payload = _read_payload(source_path)

    min_main = max(24, _coerce_int(payload.get("min_main_row_height_px"), LAYER_ROW_MIN_HEIGHT_PX))
    max_main = max(min_main, _coerce_int(payload.get("max_main_row_height_px"), 220))
    default_main = max(
        min_main,
        min(max_main, _coerce_int(payload.get("default_main_row_height_px"), LAYER_ROW_HEIGHT_PX)),
    )
    take_row = max(24, _coerce_int(payload.get("take_row_height_px"), TAKE_ROW_HEIGHT_PX))
    resize_hit_padding = max(
        2,
        _coerce_int(payload.get("resize_handle_hit_padding_px"), 5),
    )

    raw_per_kind = payload.get("layer_kind_main_row_height_px")
    per_kind_payload = raw_per_kind if isinstance(raw_per_kind, dict) else {}
    per_kind: dict[LayerKind, int] = {}
    for kind in LayerKind:
        raw_height = per_kind_payload.get(kind.value, _DEFAULT_LAYER_KIND_HEIGHTS[kind])
        resolved = max(min_main, min(max_main, _coerce_int(raw_height, default_main)))
        per_kind[kind] = resolved

    return LayerHeightConfig(
        default_main_row_height_px=default_main,
        take_row_height_px=take_row,
        min_main_row_height_px=min_main,
        max_main_row_height_px=max_main,
        resize_handle_hit_padding_px=resize_hit_padding,
        layer_kind_main_row_height_px=per_kind,
    )


@lru_cache(maxsize=1)
def timeline_layer_height_config() -> LayerHeightConfig:
    return load_layer_height_config()


__all__ = [
    "LayerHeightConfig",
    "load_layer_height_config",
    "timeline_layer_height_config",
]
