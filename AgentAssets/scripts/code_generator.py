#!/usr/bin/env python3
"""
AI-specific code generation tools for EchoZero.

Provides automated code generation following established patterns
and conventions, reducing boilerplate while maintaining quality.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BlockTemplate:
    """Template for generating new block types."""
    name: str
    block_type: str
    inputs: Dict[str, str]  # port_name -> port_type
    outputs: Dict[str, str]
    category: str
    description: str


class EchoZeroCodeGenerator:
    """Generates code following EchoZero patterns and conventions."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.templates_dir = Path(__file__).resolve().parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)

    def generate_block_processor(self, template: BlockTemplate) -> str:
        """Generate a complete block processor following EchoZero patterns."""

        # Generate input ports dict
        inputs_str = ",\n            ".join(f'"{name}": {type_}' for name, type_ in template.inputs.items())
        if inputs_str:
            inputs_str = f"\n            {inputs_str}\n        "

        # Generate output ports dict
        outputs_str = ",\n            ".join(f'"{name}": {type_}' for name, type_ in template.outputs.items())
        if outputs_str:
            outputs_str = f"\n            {outputs_str}\n        "

        code = f'''"""
{template.name} Block Processor

{template.description}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Category: {template.category}
"""

from typing import Dict, Any, Optional
from src.application.blocks.base import BlockProcessor
from src.domain.entities.data_items import DataItem
from src.domain.value_objects.port_type import PortType


class {template.name}Processor(BlockProcessor):
    """
    {template.description}

    This processor implements the {template.block_type} block functionality.
    """

    def can_process(self, block_type: str) -> bool:
        """Check if this processor can handle the given block type."""
        return block_type == "{template.block_type}"

    def get_input_ports(self) -> Dict[str, str]:
        """Define input ports for this block."""
        return {{{inputs_str}}}

    def get_output_ports(self) -> Dict[str, str]:
        """Define output ports for this block."""
        return {{{outputs_str}}}

    def process(self, inputs: Dict[str, DataItem], parameters: Dict[str, Any]) -> Dict[str, DataItem]:
        """
        Process the block inputs and generate outputs.

        Args:
            inputs: Dictionary of input data items
            parameters: Block parameters/configuration

        Returns:
            Dictionary of output data items
        """
        # TODO: Implement the actual processing logic
        # This is generated boilerplate - replace with actual implementation

        outputs = {{}}

        # Process inputs and generate outputs
        # Example structure (replace with actual logic):
        # if "input_name" in inputs:
        #     input_data = inputs["input_name"]
        #     # Process input_data...
        #     outputs["output_name"] = processed_data

        return outputs


# Auto-register this processor
from src.application.blocks.registry import BlockProcessorRegistry
BlockProcessorRegistry.register({template.name}Processor())
'''

        return code

    def generate_command_class(self, command_name: str, description: str) -> str:
        """Generate a new command class following EchoZero patterns."""

        class_name = self._to_camel_case(command_name) + "Command"

        code = f'''"""
{command_name} Command

{description}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from src.application.commands.base import EchoZeroCommand
from src.application.api.application_facade import ApplicationFacade
from src.domain.entities.result import CommandResult


class {class_name}(EchoZeroCommand):
    """
    {description}

    This command implements the {command_name.lower()} functionality.
    """

    def __init__(self, facade: ApplicationFacade, **kwargs):
        """
        Initialize the {command_name.lower()} command.

        Args:
            facade: Application facade instance
            **kwargs: Command-specific parameters
        """
        super().__init__(facade)
        # TODO: Store command parameters
        # self.parameter_name = kwargs.get('parameter_name')

    def redo(self) -> CommandResult:
        """
        Execute the {command_name.lower()} operation.

        Returns:
            CommandResult indicating success or failure
        """
        try:
            # TODO: Implement the actual command logic
            # Example:
            # result = self.facade.some_operation(self.parameter_name)
            # return CommandResult.success(f"Successfully {command_name.lower()}", data=result)

            return CommandResult.success(f"Successfully executed {command_name.lower()}")

        except Exception as e:
            return CommandResult.error(f"Failed to {command_name.lower()}: {{str(e)}}")

    def undo(self) -> CommandResult:
        """
        Undo the {command_name.lower()} operation.

        Returns:
            CommandResult indicating success or failure
        """
        try:
            # TODO: Implement the undo logic
            # This should reverse the effects of redo()

            return CommandResult.success(f"Successfully undone {command_name.lower()}")

        except Exception as e:
            return CommandResult.error(f"Failed to undo {command_name.lower()}: {{str(e)}}")
'''

        return code

    def generate_test_file(self, target_file: Path, test_type: str = "unit") -> str:
        """Generate a test file for the given target file."""

        module_name = target_file.stem
        test_name = f"test_{module_name}"

        # Determine what to test based on file content
        try:
            content = target_file.read_text()
        except FileNotFoundError:
            content = ""

        # Extract class names
        class_pattern = r'class (\w+).*?:'
        classes = re.findall(class_pattern, content)

        # Extract function names
        func_pattern = r'def (\w+)\s*\('
        functions = re.findall(func_pattern, content)

        test_functions = []
        for func in functions:
            if not func.startswith('_'):  # Skip private methods
                test_functions.append(f"def test_{func}(self):")

        for class_name in classes:
            test_functions.append(f"def test_{class_name.lower()}_creation(self):")

        test_methods = "\n    ".join([
            f"""def test_{func}(self):
        \"\"\"Test {func} functionality.\"\"\"
        # TODO: Implement test
        self.assertTrue(True)  # Placeholder assertion"""
            for func in functions if not func.startswith('_')
        ] + [
            f"""def test_{class_name.lower()}_creation(self):
        \"\"\"Test {class_name} can be created.\"\"\"
        # TODO: Implement test
        # instance = {class_name}()
        # self.assertIsInstance(instance, {class_name})"""
            for class_name in classes
        ])

        code = f'''"""
Tests for {module_name}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Type: {test_type}
"""

import unittest
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# TODO: Import the module being tested
# from src.module.path import {", ".join(classes) if classes else "ModuleClass"}


class Test{module_name.title()}(unittest.TestCase):
    """Test cases for {module_name}."""

    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize test fixtures
        pass

    def tearDown(self):
        """Clean up test fixtures."""
        # TODO: Clean up test fixtures
        pass

    {chr(10).join([f"    def test_{func}(self):{chr(10)}        \"\"\"Test {func} functionality.\"\"\"{chr(10)}        # TODO: Implement test{chr(10)}        self.assertTrue(True)  # Placeholder assertion" for func in functions if not func.startswith('_')])}

    {chr(10).join([f"    def test_{class_name.lower()}_creation(self):{chr(10)}        \"\"\"Test {class_name} can be created.\"\"\"{chr(10)}        # TODO: Implement test{chr(10)}        # instance = {class_name}(){chr(10)}        # self.assertIsInstance(instance, {class_name})" for class_name in classes])}


if __name__ == '__main__':
    unittest.main()
'''

        return code

    def generate_documentation(self, file_path: Path) -> str:
        """Generate documentation template for a file."""

        try:
            content = file_path.read_text()
        except FileNotFoundError:
            content = ""

        # Extract classes and functions
        classes = re.findall(r'class (\w+).*?:', content)
        functions = re.findall(r'def (\w+)\s*\(', content)

        doc_sections = []

        for class_name in classes:
            doc_sections.append(f"""## {class_name}

**Purpose:** [Brief description of what this class does]

**Key Methods:**
- [List important methods and their purposes]

**Usage Example:**
```python
# TODO: Add usage example
{class_name}()
```

**Dependencies:**
- [List key dependencies or relationships]
""")

        for func_name in functions:
            if not func_name.startswith('_'):
                doc_sections.append(f"""### {func_name}()

**Purpose:** [What this function does]

**Parameters:**
- [List parameters and their types/purposes]

**Returns:** [Return type and description]

**Example:**
```python
# TODO: Add example usage
result = {func_name}()
```
""")

        module_name = file_path.stem

        code = f'''# {module_name.title()} Documentation

**File:** `{file_path.name}`
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

[Brief overview of what this module does and its role in the system]

## Architecture

[Describe how this module fits into the overall architecture]

## Classes

{"".join(doc_sections)}

## Error Handling

[Describe error conditions and how they're handled]

## Testing

[Describe testing approach and key test cases]

## Future Improvements

[Note any known limitations or planned enhancements]

---
*This documentation was auto-generated. Please update with specific details.*
'''

        return code

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(word.title() for word in components)

    def create_block_from_template(self, template: BlockTemplate, output_dir: Optional[Path] = None) -> Path:
        """Create a complete block processor file from template."""

        if output_dir is None:
            output_dir = self.project_root / "src" / "application" / "blocks"

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{template.block_type.lower()}_processor.py"
        output_path = output_dir / filename

        code = self.generate_block_processor(template)

        output_path.write_text(code)
        return output_path

    def create_command_from_template(self, command_name: str, description: str,
                                   output_dir: Optional[Path] = None) -> Path:
        """Create a complete command file from template."""

        if output_dir is None:
            output_dir = self.project_root / "src" / "application" / "commands"

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{command_name.lower()}_command.py"
        output_path = output_dir / filename

        code = self.generate_command_class(command_name, description)

        output_path.write_text(code)
        return output_path


def main():
    """CLI interface for code generation."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python code_generator.py block <name> <block_type> <description>")
        print("  python code_generator.py command <name> <description>")
        print("  python code_generator.py test <target_file>")
        print("  python code_generator.py docs <target_file>")
        sys.exit(1)

    generator = EchoZeroCodeGenerator()
    command = sys.argv[1]

    if command == 'block':
        if len(sys.argv) < 5:
            print("Usage: python code_generator.py block <name> <block_type> <description>")
            sys.exit(1)

        name, block_type, description = sys.argv[2], sys.argv[3], ' '.join(sys.argv[4:])

        template = BlockTemplate(
            name=name,
            block_type=block_type,
            inputs={},  # Empty for now, can be extended
            outputs={},
            category="Generated",
            description=description
        )

        output_path = generator.create_block_from_template(template)
        print(f"Generated block processor: {output_path}")

    elif command == 'command':
        if len(sys.argv) < 4:
            print("Usage: python code_generator.py command <name> <description>")
            sys.exit(1)

        name, description = sys.argv[2], ' '.join(sys.argv[3:])

        output_path = generator.create_command_from_template(name, description)
        print(f"Generated command: {output_path}")

    elif command == 'test':
        if len(sys.argv) < 3:
            print("Usage: python code_generator.py test <target_file>")
            sys.exit(1)

        target_file = Path(sys.argv[2])
        test_code = generator.generate_test_file(target_file)

        # Write to tests directory
        test_dir = generator.project_root / "tests"
        test_file = test_dir / f"test_{target_file.stem}.py"
        test_file.write_text(test_code)
        print(f"Generated test file: {test_file}")

    elif command == 'docs':
        if len(sys.argv) < 3:
            print("Usage: python code_generator.py docs <target_file>")
            sys.exit(1)

        target_file = Path(sys.argv[2])
        docs = generator.generate_documentation(target_file)

        # Write documentation
        docs_dir = generator.project_root / "docs"
        docs_file = docs_dir / f"{target_file.stem}.md"
        docs_file.write_text(docs)
        print(f"Generated documentation: {docs_file}")


if __name__ == '__main__':
    main()

