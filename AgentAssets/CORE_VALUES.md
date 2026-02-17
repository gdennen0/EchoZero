# Core Values and Principles

## The Two Pillars

### 1. "The Best Part is No Part"

This is not about minimalism for aesthetics. It's about recognizing that every line of code, every abstraction, every feature is a liability that must earn its keep.

**What This Means:**

**In Code:**
- Question every new class, method, or abstraction
- Can we delete code instead of adding it?
- Is this abstraction solving a real problem or anticipated complexity?
- Three simple functions beat one complex abstraction

**In Features:**
- Does this solve a user problem or just seem cool?
- Can we achieve 80% of the value with 20% of the feature?
- What's the minimum that would work?
- Can existing features be combined instead?

**In Dependencies:**
- Do we really need this library?
- Can we write 50 lines instead of adding a 10MB dependency?
- What's the maintenance cost?
- What failure modes does this introduce?

**Examples:**

**Good - Removing Parts:**
```
Problem: Blocks accumulate resources
Solution: Add cleanup() method to base Block class
Result: One method, clear contract, solves entire class of problems
```

**Bad - Adding Parts:**
```
Problem: Need to format dates in CLI
Solution: Add moment.js, create DateFormatter service, add config
Result: Massive dependency for something Python datetime handles
```

### 2. "Simplicity and Refinement are Key"

Simple is not simplistic. Simple is what remains after you've removed everything unnecessary.

**What This Means:**

**In Design:**
- The interface should feel inevitable
- No surprising behavior
- Clear mental model
- One obvious way to do things

**In Implementation:**
- Code should read like prose
- No cleverness for its own sake
- Explicit over implicit
- Boring is good

**In User Experience:**
- Common tasks are easy
- Complex tasks are possible
- Errors are clear and actionable
- Progressive disclosure (simple by default, power when needed)

**Examples:**

**Good - Simple and Refined:**
```python
# ApplicationFacade method
def add_block(self, block_type: str, name: Optional[str] = None) -> CommandResult:
    # One method, clear parameters, rich return type
    # Does one thing well
```

**Bad - Complex and Scattered:**
```python
# Multiple methods for variations
def add_block(self, type): ...
def add_block_with_name(self, type, name): ...
def add_block_with_config(self, type, name, config): ...
def add_block_advanced(self, **kwargs): ...
```

## Derived Principles

### Principle: Default to Deletion

When faced with a problem, first ask: "What can I remove?"

**Questions to Ask:**
- Is this code still needed?
- Can we merge these two similar things?
- What if we just deleted this feature?
- Is this abstraction earning its keep?

**Example Decisions:**
- Remove UI positions from Block entity (backend doesn't need it)
- Remove separate Port entities (Dict[str, PortType] is simpler)
- Session-only database (delete on load, don't manage lifetime)

### Principle: Question New Abstractions

Every abstraction is a bet that complexity will grow in a specific way. Most bets lose.

**When to Abstract:**
- Pattern has emerged in 3+ places (Rule of Three)
- Abstraction is simpler than duplication
- Abstraction has single, clear purpose
- Cost of being wrong is low

**When NOT to Abstract:**
- "We might need this someday"
- "This is how big systems do it"
- "It's more elegant"
- Making one thing "more general"

**Example:**
```python
# Don't create BlockPortManager, PortValidator, PortTypeRegistry, etc.
# Just use Dict[str, PortType] on Block
```

### Principle: Make Common Things Easy, Complex Things Possible

Optimize for the 80% use case. Don't make simple things complex to enable edge cases.

**In API Design:**
```python
# Common case: simple
facade.add_block("LoadAudio")

# With customization: still straightforward
facade.add_block("LoadAudio", name="MyLoader")

# Don't force:
facade.add_block(
    BlockConfiguration(
        type=BlockType.LOAD_AUDIO,
        name=Optional("MyLoader"),
        metadata=MetadataConfig(...)
    )
)
```

### Principle: Explicit Over Implicit

Magic is for entertainment, not software. Be boring and clear.

**Prefer:**
```python
def execute_project(self, project_id: str) -> ExecutionResult:
    blocks = self.get_sorted_blocks(project_id)
    for block in blocks:
        self.execute_block(block)
```

**Over:**
```python
@auto_execute
@with_validation
@event_driven
def process(self):
    # What's happening? Who knows!
```

**Why:**
- Easy to debug (just read the code)
- Easy to test (no hidden behavior)
- Easy to modify (no surprising side effects)
- Easy to understand (no magic to learn)

### Principle: Favor Composition Over Inheritance

Build complex behavior from simple, independent parts.

**Prefer:**
```python
class ExecutionEngine:
    def __init__(self, validator, sorter, event_bus):
        self.validator = validator
        self.sorter = sorter
        self.event_bus = event_bus
    
    def execute(self, project_id):
        if not self.validator.validate(project_id):
            return ValidationError()
        blocks = self.sorter.sort(project_id)
        self.event_bus.publish(ExecutionStarted())
        # ...
```

**Over:**
```python
class ExecutionEngine(Validator, Sorter, EventPublisher, LogManager, ErrorHandler):
    # Deep inheritance hierarchy, tight coupling
```

### Principle: Optimize for Reading, Not Writing

Code is read 10x more than it's written. Optimize for the reader.

**Good:**
```python
def add_block(self, block_type: str, name: Optional[str] = None) -> CommandResult:
    """Add a block to the current project"""
    if not self.current_project:
        return CommandResult.error("No project loaded")
    
    if not self.block_type_exists(block_type):
        return CommandResult.error(f"Unknown block type: {block_type}")
    
    block = self.block_service.add_block(
        project_id=self.current_project.id,
        block_type=block_type,
        name=name
    )
    
    return CommandResult.success(f"Added block '{block.name}'", data=block)
```

**Bad:**
```python
def add_block(self, t, n=None):
    return self.bs.add(self.cp.id, t, n) if self.cp and self.bte(t) else None
```

### Principle: Errors Should Be Impossible or Obvious

Design APIs that either prevent errors or make them immediately visible.

**Prevent Errors:**
```python
# Type system prevents wrong port types
def add_input_port(self, name: str, port_type: PortType):
    # Can only pass PortType, not string
```

**Make Errors Obvious:**
```python
# Rich error returns
CommandResult.error(
    message="Cannot connect blocks",
    errors=[
        "Port 'audio' on block 'LoadAudio1' is type 'Audio'",
        "Port 'events' on block 'DetectOnsets1' is type 'Event'",
        "Types must match"
    ]
)
```

### Principle: Data Over Code

Prefer data-driven solutions. Data is easier to understand and modify than code.

**Good:**
```python
# Block definitions in registry (data)
registry.register(BlockTypeMetadata(
    name="Load Audio",
    type_id="LoadAudio",
    inputs={},
    outputs={"audio": AUDIO_TYPE}
))
```

**Bad:**
```python
# Block definitions in class hierarchy (code)
class LoadAudioBlock(AudioBlock, InputBlock, FileHandlingBlock):
    def get_output_ports(self):
        return [Port("audio", AUDIO_TYPE)]
```

## Anti-Patterns to Avoid

### Anti-Pattern: Premature Generalization

Don't build for imagined future requirements.

**Symptoms:**
- "We might need to support X someday"
- "This makes it more flexible"
- "It's more professional/enterprise-grade"

**Cure:**
- Build for current requirements
- Refactor when patterns emerge
- YAGNI (You Ain't Gonna Need It)

### Anti-Pattern: Resume-Driven Development

Don't use cool tech because it looks good on a resume.

**Symptoms:**
- "Let's use GraphQL/gRPC/Kubernetes"
- "Everyone uses microservices"
- "This is the modern way"

**Cure:**
- Use boring, proven technology
- Pick tools that solve your actual problems
- Optimize for maintainability

### Anti-Pattern: Not Invented Here

Don't rewrite everything from scratch, but don't add dependencies thoughtlessly.

**Balance:**
- Use libraries for complex domains (audio processing, ML)
- Write your own for simple, core logic (validation, formatting)
- Consider maintenance cost of dependencies

### Anti-Pattern: Abstraction Addiction

More abstraction layers do not make better software.

**Symptoms:**
- Interfaces with one implementation
- Factories that create one type
- Managers/Handlers/Helpers/Utils everywhere
- Can't find where code actually executes

**Cure:**
- Start concrete, abstract when needed
- Each abstraction must solve real duplication
- Keep call chains short and traceable

### Anti-Pattern: Feature Creep

Every feature adds complexity. Most features are used rarely or never.

**Symptoms:**
- "It would be cool if..."
- "Users might want..."
- "While we're at it..."

**Cure:**
- Require evidence of user need
- Build minimum, iterate based on usage
- Say no by default

## Decision Framework

When evaluating any proposal, apply these filters in order:

### Filter 1: Is This Necessary?

**Ask:**
- What problem does this solve?
- Do users actually have this problem?
- Can we just not do this?

**If answer unclear:** Reject or defer

### Filter 2: Can We Remove Instead?

**Ask:**
- Can we solve this by removing something?
- Is there existing complexity this eliminates?
- What's the net complexity change?

**If net complexity increases significantly:** Reconsider

### Filter 3: What's the Simplest Solution?

**Ask:**
- What's the dumbest thing that could work?
- Can we do this with existing tools?
- What's the 20% that gives 80% value?

**Always start with simplest:** Elaborate only if necessary

### Filter 4: What's the Cost?

**Ask:**
- Lines of code added?
- New dependencies?
- New concepts users must learn?
- Testing difficulty?
- Maintenance burden?

**If cost > benefit:** Reject or simplify

### Filter 5: Is It Reversible?

**Ask:**
- Can we undo this decision later?
- Does this commit us to a path?
- What's the cost of being wrong?

**Prefer reversible decisions:** Lower risk

## Examples in Practice

### Case Study 1: Memory Leak Fix

**Problem:** Blocks accumulate resources (timers, media players, UI windows)

**Bad Solution:**
- Create ResourceManager class
- Add ResourceRegistry
- Implement automatic lifecycle tracking
- Add configuration for resource limits
- 500+ lines of new code

**Good Solution:**
- Add `cleanup()` method to Block base class
- Call cleanup when removing blocks or unloading project
- Implement specific cleanup in blocks that need it
- 50 lines of code, clear contract

**Why Good:**
- Simple base class method
- Explicit (call cleanup when done)
- Extensible (override as needed)
- No new abstractions
- Easy to understand and debug

### Case Study 2: Project File Format

**Problem:** Need to persist projects

**Bad Solution:**
- Design custom binary format
- Implement compression
- Add encryption support
- Create migration system
- Build custom serializer

**Good Solution:**
- Use JSON
- Store as .ez file
- Version field for future migrations
- Done

**Why Good:**
- Human-readable (easy to debug)
- Version control friendly
- No custom code needed
- Portable
- Simple

### Case Study 3: ApplicationFacade

**Problem:** CommandParser too large, GUI would duplicate logic

**Bad Solution:**
- Keep CommandParser as-is
- Create separate GUICommandParser
- Duplicate validation in both
- Hope they stay in sync

**Good Solution:**
- Extract ApplicationFacade
- CommandParser becomes thin adapter
- All logic in Facade
- GUI uses same Facade
- Rich CommandResult type

**Why Good:**
- Eliminates duplication before it exists
- Single source of truth
- Easier to test
- Enables future GUI with no extra work
- Actually reduced code size

## Mantras for Daily Development

1. **"Can I delete this instead?"**
2. **"What's the simplest thing that could work?"**
3. **"Will I be happy debugging this at 3am?"**
4. **"Would I understand this if I read it in 6 months?"**
5. **"Is this solving a real problem or an imagined one?"**
6. **"Am I being clever or clear?"**
7. **"Does this make the system simpler or more complex?"**
8. **"The best part is no part"**

## Remember

Simplicity is not about doing less work. It's about doing the right work.

Removing code is harder than adding it. Saying no is harder than saying yes. Keeping things simple requires constant discipline.

But simple systems are:
- Easier to understand
- Easier to modify
- Easier to debug
- Easier to test
- Easier to maintain
- More reliable
- More performant

**Complexity is the default. Simplicity requires intention.**

The Council's job is to be that intention - to push back against the natural tendency toward complexity, to advocate for removal, to demand justification for additions.

Be the guardians of simplicity.

**The best part is no part.**

