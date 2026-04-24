"""Shared settings-page contracts for reusable EchoZero settings surfaces.
Exists to keep field metadata typed and reusable outside any single feature area.
Connects settings-owning services to neutral Qt rendering without widget-local semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SettingsFieldWidget(str, Enum):
    """Supported widget shapes for neutral settings rendering."""

    TEXT = "text"
    DROPDOWN = "dropdown"
    TOGGLE = "toggle"
    NUMBER = "number"


class SettingsFieldSurface(str, Enum):
    """Visibility level for one settings field."""

    PRIMARY = "primary"
    ADVANCED = "advanced"
    HIDDEN = "hidden"


@dataclass(slots=True, frozen=True)
class SettingsOption:
    """One selectable option for a dropdown-backed settings field."""

    value: object
    label: str


@dataclass(slots=True, frozen=True)
class SettingsField:
    """Typed presentation contract for one reusable settings field."""

    key: str
    label: str
    value: object
    default_value: object
    persisted_value: object | None = None
    is_dirty: bool = False
    widget: SettingsFieldWidget = SettingsFieldWidget.TEXT
    description: str = ""
    enabled: bool = True
    surface: SettingsFieldSurface = SettingsFieldSurface.PRIMARY
    placeholder: str = ""
    units: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[SettingsOption, ...] = ()


@dataclass(slots=True, frozen=True)
class SettingsSection:
    """One labeled group of related settings fields."""

    key: str
    title: str
    description: str = ""
    fields: tuple[SettingsField, ...] = ()


@dataclass(slots=True, frozen=True)
class SettingsPage:
    """Rendered settings page contract for one bounded settings surface."""

    key: str
    title: str
    summary: str = ""
    sections: tuple[SettingsSection, ...] = ()
    warnings: tuple[str, ...] = ()
