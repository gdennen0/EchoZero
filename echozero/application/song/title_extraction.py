"""Song import title extraction helpers for filename-driven imports.
Exists to turn operator file naming conventions into cleaner default song titles.
Connects machine-local song import preferences to the canonical add-song import paths.
"""

from __future__ import annotations

from pathlib import Path
import re

from echozero.application.settings.models import SongImportNameMode

_TITLE_SPLIT_PATTERN = re.compile(r"[ _-]+")
_VERSION_TOKEN_PATTERN = re.compile(r"^(?:v|ver|version)\d+$", re.IGNORECASE)
_MEASURED_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:bpm|fps)$", re.IGNORECASE)
_NUMBER_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
_HOUR_TOKEN_PATTERN = re.compile(r"^\d+(?:h|hr|hrs|hour|hours)$", re.IGNORECASE)
_METADATA_TOKENS = frozenset(
    {
        "df",
        "drop",
        "fps",
        "frame",
        "framerate",
        "guide",
        "hour",
        "hours",
        "ltc",
        "mix",
        "ndf",
        "ref",
        "refs",
        "smpte",
        "tc",
        "timecode",
    }
)


def resolve_import_song_titles(
    audio_paths: tuple[str, ...],
    *,
    name_mode: SongImportNameMode,
) -> dict[str, str]:
    """Resolve display-ready song titles for one import batch."""

    default_titles = {
        audio_path: (Path(audio_path).stem.strip() or "Imported Song") for audio_path in audio_paths
    }
    if name_mode is not SongImportNameMode.EXTRACT_TITLE:
        return default_titles

    tokenized = {
        audio_path: _strip_trailing_metadata(_tokenize_title(Path(audio_path).stem))
        for audio_path in audio_paths
    }
    prefix_tokens = _shared_artist_prefix(tuple(tokenized.values()))
    resolved: dict[str, str] = {}
    for audio_path, tokens in tokenized.items():
        title_tokens = list(tokens)
        if prefix_tokens and len(title_tokens) > len(prefix_tokens):
            title_tokens = title_tokens[len(prefix_tokens) :]
        elif len(title_tokens) >= 2 and _looks_like_artist_prefix(title_tokens[0]):
            title_tokens = title_tokens[1:]
        resolved[audio_path] = _humanize_title_tokens(title_tokens) or default_titles[audio_path]
    return resolved


def _tokenize_title(stem: str) -> list[str]:
    return [token for token in _TITLE_SPLIT_PATTERN.split(stem.strip()) if token]


def _strip_trailing_metadata(tokens: list[str]) -> list[str]:
    resolved = list(tokens)
    while resolved:
        candidate = resolved[-1]
        if _is_metadata_token(candidate):
            resolved.pop()
            continue
        if len(resolved) >= 2 and _NUMBER_TOKEN_PATTERN.match(candidate):
            prior = resolved[-2].casefold()
            if prior in _METADATA_TOKENS:
                resolved.pop()
                continue
        break
    return resolved


def _is_metadata_token(token: str) -> bool:
    normalized = token.strip().casefold()
    if not normalized:
        return True
    if normalized in _METADATA_TOKENS:
        return True
    if _VERSION_TOKEN_PATTERN.match(normalized):
        return True
    if _MEASURED_TOKEN_PATTERN.match(normalized):
        return True
    if _HOUR_TOKEN_PATTERN.match(normalized):
        return True
    return False


def _shared_artist_prefix(token_groups: tuple[list[str], ...]) -> tuple[str, ...]:
    if len(token_groups) < 2 or any(len(tokens) < 2 for tokens in token_groups):
        return ()
    prefix: list[str] = []
    shortest = min(len(tokens) for tokens in token_groups)
    for index in range(shortest):
        candidate = token_groups[0][index]
        if any(tokens[index] != candidate for tokens in token_groups[1:]):
            break
        prefix.append(candidate)
    if not prefix:
        return ()
    if not _looks_like_artist_prefix("".join(prefix)):
        return ()
    if any(len(tokens) <= len(prefix) for tokens in token_groups):
        return ()
    return tuple(prefix)


def _looks_like_artist_prefix(token: str) -> bool:
    normalized = token.strip()
    if len(normalized) < 3 or any(character.isdigit() for character in normalized):
        return False
    upper_count = sum(1 for character in normalized if character.isupper())
    return upper_count >= 2 or len(normalized) >= 8


def _humanize_title_tokens(tokens: list[str]) -> str:
    formatted = [_humanize_title_token(token) for token in tokens if token.strip()]
    return " ".join(token for token in formatted if token).strip()


def _humanize_title_token(token: str) -> str:
    text = token.replace(".", " ")
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[a-z0-9'])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()
