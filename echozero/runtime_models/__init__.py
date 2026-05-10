"""
Shared runtime model layer for Foundry exports and EchoZero inference.
Exists because training and app inference must share stable runtime-safe architectures and bundle loading.
Used by Foundry trainers, app processors, and future model install/selection services.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .architectures import CrnnRuntimeModel, SimpleCnnRuntimeModel
    from .bundle_compat import backfill_manifest_fingerprint, upgrade_installed_runtime_bundles
    from .loader import LoadedRuntimeModel, load_runtime_model

__all__ = [
    "CrnnRuntimeModel",
    "SimpleCnnRuntimeModel",
    "LoadedRuntimeModel",
    "backfill_manifest_fingerprint",
    "load_runtime_model",
    "upgrade_installed_runtime_bundles",
]

_LAZY_EXPORTS = {
    "CrnnRuntimeModel": ("echozero.runtime_models.architectures", "CrnnRuntimeModel"),
    "SimpleCnnRuntimeModel": ("echozero.runtime_models.architectures", "SimpleCnnRuntimeModel"),
    "LoadedRuntimeModel": ("echozero.runtime_models.loader", "LoadedRuntimeModel"),
    "backfill_manifest_fingerprint": (
        "echozero.runtime_models.bundle_compat",
        "backfill_manifest_fingerprint",
    ),
    "load_runtime_model": ("echozero.runtime_models.loader", "load_runtime_model"),
    "upgrade_installed_runtime_bundles": (
        "echozero.runtime_models.bundle_compat",
        "upgrade_installed_runtime_bundles",
    ),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)
