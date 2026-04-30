from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact_service import ArtifactService
    from .baseline_trainer import BaselineTrainer
    from .cnn_trainer import CnnTrainer
    from .crnn_trainer import CrnnTrainer
    from .dataset_service import DatasetService
    from .eval_service import EvalService
    from .query_service import FoundryQueryService
    from .review_session_service import ReviewSessionService
    from .review_extraction_service import ReviewExtractionService
    from .review_commit_mapper import (
        build_explicit_commit_from_item,
        build_review_commit_command,
        build_review_commit_context,
        normalize_source_provenance,
    )
    from .review_pipeline_controller import ReviewPipelineController
    from .review_signal_service import ReviewSignalService
    from .runtime_bundle_install_service import RuntimeBundleInstallService
    from .run_notification_service import RunNotificationService
    from .run_spec_validator import RunSpecValidator
    from .run_telemetry_service import RunTelemetryService
    from .split_balance_service import SplitBalanceService
    from .train_run_service import TrainRunService
    from .training_orchestrator import TrainingOrchestrator
    from .trainer_backend_factory import TrainerBackendFactory
    from .training_runtime import TrainingNumericsError

__all__ = [
    "ArtifactService",
    "BaselineTrainer",
    "DatasetService",
    "EvalService",
    "CnnTrainer",
    "CrnnTrainer",
    "FoundryQueryService",
    "ReviewSessionService",
    "ReviewExtractionService",
    "build_explicit_commit_from_item",
    "build_review_commit_command",
    "build_review_commit_context",
    "normalize_source_provenance",
    "ReviewPipelineController",
    "ReviewSignalService",
    "RuntimeBundleInstallService",
    "RunNotificationService",
    "RunSpecValidator",
    "RunTelemetryService",
    "SplitBalanceService",
    "TrainRunService",
    "TrainingOrchestrator",
    "TrainerBackendFactory",
    "TrainingNumericsError",
]

_LAZY_EXPORTS = {
    "ArtifactService": ("echozero.foundry.services.artifact_service", "ArtifactService"),
    "BaselineTrainer": ("echozero.foundry.services.baseline_trainer", "BaselineTrainer"),
    "DatasetService": ("echozero.foundry.services.dataset_service", "DatasetService"),
    "EvalService": ("echozero.foundry.services.eval_service", "EvalService"),
    "CnnTrainer": ("echozero.foundry.services.cnn_trainer", "CnnTrainer"),
    "CrnnTrainer": ("echozero.foundry.services.crnn_trainer", "CrnnTrainer"),
    "FoundryQueryService": ("echozero.foundry.services.query_service", "FoundryQueryService"),
    "ReviewSessionService": ("echozero.foundry.services.review_session_service", "ReviewSessionService"),
    "ReviewExtractionService": (
        "echozero.foundry.services.review_extraction_service",
        "ReviewExtractionService",
    ),
    "build_explicit_commit_from_item": (
        "echozero.foundry.services.review_commit_mapper",
        "build_explicit_commit_from_item",
    ),
    "build_review_commit_command": (
        "echozero.foundry.services.review_commit_mapper",
        "build_review_commit_command",
    ),
    "build_review_commit_context": (
        "echozero.foundry.services.review_commit_mapper",
        "build_review_commit_context",
    ),
    "normalize_source_provenance": (
        "echozero.foundry.services.review_commit_mapper",
        "normalize_source_provenance",
    ),
    "ReviewPipelineController": (
        "echozero.foundry.services.review_pipeline_controller",
        "ReviewPipelineController",
    ),
    "ReviewSignalService": ("echozero.foundry.services.review_signal_service", "ReviewSignalService"),
    "RuntimeBundleInstallService": (
        "echozero.foundry.services.runtime_bundle_install_service",
        "RuntimeBundleInstallService",
    ),
    "RunNotificationService": ("echozero.foundry.services.run_notification_service", "RunNotificationService"),
    "RunSpecValidator": ("echozero.foundry.services.run_spec_validator", "RunSpecValidator"),
    "RunTelemetryService": ("echozero.foundry.services.run_telemetry_service", "RunTelemetryService"),
    "SplitBalanceService": ("echozero.foundry.services.split_balance_service", "SplitBalanceService"),
    "TrainRunService": ("echozero.foundry.services.train_run_service", "TrainRunService"),
    "TrainingOrchestrator": ("echozero.foundry.services.training_orchestrator", "TrainingOrchestrator"),
    "TrainerBackendFactory": ("echozero.foundry.services.trainer_backend_factory", "TrainerBackendFactory"),
    "TrainingNumericsError": ("echozero.foundry.services.training_runtime", "TrainingNumericsError"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)
