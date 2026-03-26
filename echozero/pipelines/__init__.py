"""
Pipeline template infrastructure: Registry, decorators, and template definitions.
Exists because pipeline templates are Python builder functions that return Graphs,
replacing EZ1's imperative action sequences with declarative, testable, versionable code.
"""

from echozero.pipelines.registry import PipelineRegistry, PromotedParam, apply_bindings, pipeline_template
