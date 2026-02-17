---
name: echozero-block-execution
description: Block execution engine and execution flow in EchoZero. Use when understanding how blocks execute, topological sort, data flow, execution order, or when debugging execution issues.
---

# Block Execution

## Execution Engine

`BlockExecutionEngine` in `src/features/execution/application/execution_engine.py`

Responsibilities:
- Topological sort of blocks by connection dependencies
- Data flow: pull upstream outputs, pass as inputs
- Error handling, progress events
- Data item persistence

## Execution Order

Uses Kahn's algorithm via `topological_sort_blocks()` in `src/features/execution/application/topological_sort.py`:

1. Blocks with no incoming connections execute first
2. After block executes, dependents become available
3. Continues until all blocks executed
4. Raises `CyclicDependencyError` if circular dependencies

## 5-Step Processor Flow

For each block, the engine:

1. **step_clear_local_data()** - Clear owned data items
2. **Pull upstream data** - Engine gathers from connected source blocks
3. **step_pre_process()** - Pre-processing hook (optional)
4. **process()** - Main logic (required)
5. **step_post_process()** - Post-processing hook (optional)

## Input Gathering

Engine maps connections to build inputs dict:

```python
# (target_block_id, target_input_name) -> (source_block_id, source_output_name)
# Retrieve DataItem from source block's output
# Pass as inputs[input_name] = DataItem
```

Multiple connections to same input: merged (e.g., multiple EventDataItems for Event ports).

## Key Components

- ExecutionEngine: `src/features/execution/application/execution_engine.py`
- Topological sort: `src/features/execution/application/topological_sort.py`
- BlockProcessor: `src/application/processing/block_processor.py`
- Execution events: ExecutionStarted, BlockExecuted, BlockExecutionFailed, ExecutionCompleted

## ExecutionResult

```python
@dataclass
class ExecutionResult:
    success: bool
    executed_blocks: List[str]
    failed_blocks: List[str]
    output_data: Dict[str, Dict[str, DataItem]]
    errors: Dict[str, str]
```

## Triggering Execution

- `facade.execute_project(project_id)` - Execute all blocks in project
- Setlist processing runs execution per song with action pipeline
