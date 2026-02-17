#!/usr/bin/env python3
"""
EchoZero Helper for Cursor AI Agents

This script provides a simple interface for AI agents to get EchoZero context
when working in Cursor IDE. Just type your request and get intelligent context.
"""

import sys
import json
import subprocess
from pathlib import Path


def get_echozero_context(user_input: str, agent_id: str = "cursor_ai") -> str:
    """
    Get EchoZero context for AI agent assistance.

    Args:
        user_input: The user's natural language request
        agent_id: Identifier for the AI agent (claude, gpt, etc.)

    Returns:
        Formatted context string ready for AI agent consumption
    """
    try:
        # Determine context type and domain from input
        context_type, domain = analyze_input(user_input)

        # Build the context command
        cmd = [
            sys.executable,
            "AgentAssets/scripts/context_cli.py",
            "context",
            agent_id,
            user_input
        ]

        if domain:
            cmd.extend(["--domain", domain])

        # Run the context command
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            context_data = json.loads(result.stdout)

            # Format for AI agent consumption
            return format_context_for_agent(context_data, user_input)
        else:
            return f"Error getting context: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"


def analyze_input(user_input: str) -> tuple[str, str]:
    """Analyze user input to determine context type and domain."""
    input_lower = user_input.lower()

    # Determine context type
    if any(word in input_lower for word in ["block", "create", "implement", "processor"]):
        context_type = "block_creation"
    elif any(word in input_lower for word in ["ui", "interface", "widget", "gui"]):
        context_type = "ui_development"
    elif any(word in input_lower for word in ["command", "undo", "redo", "action"]):
        context_type = "command_creation"
    elif any(word in input_lower for word in ["test", "testing", "pytest"]):
        context_type = "testing"
    else:
        context_type = "general"

    # Determine domain
    if any(word in input_lower for word in ["audio", "sound", "music", "wave"]):
        domain = "audio"
    elif any(word in input_lower for word in ["block", "processor", "workflow"]):
        domain = "blocks"
    elif any(word in input_lower for word in ["ui", "interface", "display"]):
        domain = "ui"
    else:
        domain = "general"

    return context_type, domain


def format_context_for_agent(context_data: dict, original_request: str) -> str:
    """Format context data into a readable format for AI agents."""
    output = []

    output.append("## 🎯 EchoZero Context for Your Request")
    output.append(f"**Original Request:** {original_request}")
    output.append("")

    # Core Values (always include)
    core_context = context_data.get("core_context", {})
    if "core_values" in core_context:
        output.append("## 📋 Core EchoZero Values")
        values = core_context["core_values"]
        if "best_part_is_no_part" in values:
            output.append("**Best Part is No Part:** " + values["best_part_is_no_part"]["principle"])
        if "simplicity_and_refinement" in values:
            output.append("**Simplicity & Refinement:** " + values["simplicity_and_refinement"]["principle"])
        output.append("")

    # Architectural Patterns
    if "architectural_patterns" in core_context:
        output.append("## 🏗️ Architectural Patterns")
        patterns = core_context["architectural_patterns"]
        if "key_patterns" in patterns:
            for pattern, desc in patterns["key_patterns"].items():
                output.append(f"- **{pattern}:** {desc}")
        output.append("")

    # Domain Knowledge
    if "domain_knowledge" in core_context and core_context["domain_knowledge"]:
        output.append("## 🎨 Domain Knowledge")
        for domain, knowledge in core_context["domain_knowledge"].items():
            output.append(f"**{domain.title()}:** {str(knowledge)[:200]}...")
        output.append("")

    # Quality Guidelines
    if "quality_guidelines" in core_context:
        output.append("## ✅ Quality Guidelines")
        guidelines = core_context["quality_guidelines"]
        if "code_quality" in guidelines:
            output.append("**Code Quality:**")
            for item, desc in guidelines["code_quality"].items():
                output.append(f"- {item.replace('_', ' ').title()}: {desc}")
        output.append("")

    # Current Capabilities
    if "current_capabilities" in core_context:
        output.append("## 🚀 Current System Capabilities")
        caps = core_context["current_capabilities"]
        if "available_blocks" in caps:
            output.append(f"**Available Blocks:** {', '.join(caps['available_blocks'])}")
        if "supported_formats" in caps:
            output.append(f"**Supported Formats:** {', '.join(caps['supported_formats'])}")
        output.append("")

    # Query-specific guidance
    if "query_specific" in core_context:
        output.append("## 🎯 Specific Guidance for Your Task")
        query_info = core_context["query_specific"]
        for key, value in query_info.items():
            if isinstance(value, dict):
                output.append(f"**{key.replace('_', ' ').title()}:**")
                for sub_key, sub_value in value.items():
                    output.append(f"- {sub_key.replace('_', ' ').title()}: {sub_value}")
            else:
                output.append(f"**{key.replace('_', ' ').title()}:** {value}")
        output.append("")

    # Agent-specific context
    agent_context = context_data.get("agent_specific_context", {})
    if agent_context:
        output.append("## 🤖 Agent-Specific Context")
        if "successful_patterns" in agent_context and agent_context["successful_patterns"]:
            output.append(f"**Your Successful Patterns:** {', '.join(agent_context['successful_patterns'])}")
        if "preferred_types" in agent_context and agent_context["preferred_types"]:
            output.append("**Your Preferences:**")
            for pref_type, score in agent_context["preferred_types"].items():
                output.append(f"- {pref_type}: {score}")
        output.append("")

    # Learning insights
    learning = context_data.get("learning_insights", {})
    if learning and "agent_learning" in learning:
        agent_learning = learning["agent_learning"]
        if agent_learning.get("effectiveness_trend"):
            output.append(f"## 📈 Your Learning Trend: {agent_learning['effectiveness_trend']}")
            output.append("")

    # Final instructions
    output.append("## 📝 Implementation Instructions")
    output.append("1. **Follow the BlockProcessor interface** exactly as shown in existing blocks")
    output.append("2. **Add auto-registration** at the end of your file")
    output.append("3. **Include comprehensive error handling** with ProcessingError")
    output.append("4. **Add full documentation** with type hints")
    output.append("5. **Create corresponding tests** in the tests directory")
    output.append("")
    output.append("**Remember:** Question every addition, prefer existing patterns, keep it simple!")

    return "\n".join(output)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python echozero_helper.py \"Your request here\" [agent_id]")
        print("Example: python echozero_helper.py \"Make a new block to do XYZ\" claude-3")
        sys.exit(1)

    user_request = sys.argv[1]
    agent_id = sys.argv[2] if len(sys.argv) > 2 else "cursor_ai"

    context = get_echozero_context(user_request, agent_id)
    print(context)


if __name__ == "__main__":
    main()

