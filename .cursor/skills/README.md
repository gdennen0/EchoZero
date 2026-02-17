# EchoZero Cursor Skills

Project skills provide EchoZero-specific context for AI agents. Skills are applied automatically based on user queries.

## Skill Index

| Skill | Use When |
|-------|----------|
| **echozero-core-values** | Evaluating proposals, refactoring, design decisions |
| **echozero-project-overview** | Navigating codebase, finding where to add code |
| **echozero-block-implementation** | Creating blocks, BlockProcessor, quick actions |
| **echozero-connecting-blocks** | Connecting blocks, port compatibility, data flow |
| **echozero-block-execution** | Execution engine, topological sort, data flow |
| **echozero-undo-redo** | Undoable commands, EchoZeroCommand, command bus |
| **echozero-progress-tracking** | Progress bars, long-running operations |
| **echozero-settings-abstraction** | Block settings, BlockSettingsManager |
| **echozero-block-panel** | Block panel creation, BlockPanelBase |
| **echozero-setlist-actions** | Setlist actions, quick actions, action sets |
| **echozero-refactor** | Refactoring evaluation, @refactor |
| **echozero-cleanup** | Block cleanup, resource management |
| **echozero-ma3-integration** | MA3 Lua, DataPool, OSC, hooks, show manager |
| **echozero-ui-components** | Qt dialogs, design system, quick action input |
| **echozero-council** | Major proposals, council review |
| **echozero-commands** | @feature, @refactor, @cleanup, @research workflows |
| **echozero-demucs** | SeparatorBlock, audio source separation |

## Most Used (Tier 1)

- **echozero-block-implementation** - Creating blocks
- **echozero-block-panel** - Block configuration UI
- **echozero-setlist-actions** - Quick actions, action sets
- **echozero-core-values** - Design decisions
- **echozero-settings-abstraction** - Block configurable settings

## Migration from AgentAssets

These skills consolidate content from `AgentAssets/`. AgentAssets remains the source for detailed guides; skills provide concise, discoverable summaries.

Reference paths in skills point to AgentAssets for full documentation.
