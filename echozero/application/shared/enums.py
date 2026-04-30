"""Shared enums used across the new EchoZero application layer."""

from enum import Enum


class LayerKind(str, Enum):
    AUDIO = "audio"
    EVENT = "event"
    SECTION = "section"
    AUTOMATION = "automation"
    REFERENCE = "reference"
    GROUP = "group"
    SYNC = "sync"


class PlaybackMode(str, Enum):
    NONE = "none"
    CONTINUOUS_AUDIO = "continuous_audio"
    EVENT_TONE = "event_tone"
    EVENT_SLICE = "event_slice"


class SyncMode(str, Enum):
    NONE = "none"
    INTERNAL = "internal"
    EXTERNAL = "external"
    MA3 = "ma3"


class FollowMode(str, Enum):
    OFF = "off"
    PAGE = "page"
    SMOOTH = "smooth"
    CENTER = "center"


class PlaybackStatus(str, Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"
