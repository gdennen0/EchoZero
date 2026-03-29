"""
Editor infrastructure: CQRS command dispatch, reactive coordinator, execution cache.

This package is for the FUTURE visual node editor (Stage Zero Editor).
It provides:
- Command-based graph mutation with undo/redo support
- Reactive coordinator that auto-re-executes on changes
- Execution cache for incremental re-computation

NOT USED by the current pipeline builder (echozero.pipelines.pipeline)
or the Orchestrator (echozero.services.orchestrator). Those use the
engine directly via Pipeline.add() and ExecutionEngine.run().

When building the visual editor, import from here:
    from echozero.editor.pipeline import Pipeline as EditorPipeline
    from echozero.editor.coordinator import Coordinator
"""
