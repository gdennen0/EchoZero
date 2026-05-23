"""Foundry model evolution service boundary.
Exists so user-fixed Events can become event-span training data without one-off scripts.
Connects review truth, sample materialization, lineage, and candidate run planning.
"""

from echozero.foundry.model_evolution.lineage import ModelLineage, ModelLineageResolver
from echozero.foundry.model_evolution.planner import (
    CandidateModelPlan,
    EvolutionTrainingProfile,
    ModelEvolutionPlanner,
)
from echozero.foundry.model_evolution.sample_materializer import (
    MaterializedRuntimeSample,
    RuntimeWindowMaterializer,
    RuntimeWindowPolicy,
)
from echozero.foundry.model_evolution.service import (
    ModelEvolutionRunRequest,
    ModelEvolutionRunResult,
    ModelEvolutionService,
)
from echozero.foundry.model_evolution.truth import FixedEventTruth

__all__ = [
    "CandidateModelPlan",
    "EvolutionTrainingProfile",
    "FixedEventTruth",
    "MaterializedRuntimeSample",
    "ModelEvolutionPlanner",
    "ModelEvolutionRunRequest",
    "ModelEvolutionRunResult",
    "ModelEvolutionService",
    "ModelLineage",
    "ModelLineageResolver",
    "RuntimeWindowMaterializer",
    "RuntimeWindowPolicy",
]
