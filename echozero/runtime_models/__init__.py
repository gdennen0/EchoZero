"""
Shared runtime model layer for Foundry exports and EchoZero inference.
Exists because training and app inference must share stable runtime-safe architectures and bundle loading.
Used by Foundry trainers, app processors, and future model install/selection services.
"""

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
