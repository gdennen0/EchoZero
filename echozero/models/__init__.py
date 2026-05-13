"""Model distribution and runtime-model catalogs for EchoZero.
Exists to keep app-installed model assets separate from packaged binaries.
Connects central registry manifests, local indexes, and legacy model catalog compatibility.
"""

from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelStatus,
    ModelType,
)
from echozero.models.distribution import (
    InstalledModelRecord,
    ModelInstallState,
    RegistryModelListing,
    RegistryModelEntry,
    default_registry_manifest_source,
    discover_registry_models,
    install_model_from_registry,
    list_installed_models,
    save_registry_manifest_source,
)

__all__ = [
    "InstalledModelRecord",
    "ModelInstallState",
    "ModelCard",
    "ModelRegistry",
    "ModelSource",
    "ModelStatus",
    "ModelType",
    "RegistryModelEntry",
    "RegistryModelListing",
    "default_registry_manifest_source",
    "discover_registry_models",
    "install_model_from_registry",
    "list_installed_models",
    "save_registry_manifest_source",
]
