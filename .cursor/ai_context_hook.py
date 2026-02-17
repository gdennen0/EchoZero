#!/usr/bin/env python3
"""
AI Context Hook for Cursor IDE

Automatically provides EchoZero context to AI agents without manual commands.
This hook intercepts AI agent interactions and injects relevant context.
"""

import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List


class AICursorContextHook:
    """Hook that automatically provides context to AI agents in Cursor."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.agent_assets_root = self.project_root / "AgentAssets"
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 300  # 5 minutes

        # Keywords that trigger EchoZero context
        self.echozero_keywords = {
            'block', 'echozero', 'processor', 'audio', 'processing',
            'command', 'ui', 'interface', 'test', 'testing', 'implement',
            'create', 'add', 'new', 'component', 'widget', 'panel',
            'functionality', 'feature', 'code', 'development'
        }

        # AI agent identifiers
        self.agent_identifiers = {
            'claude', 'gpt', 'gemini', 'bard', 'assistant', 'ai', 'bot'
        }

    def should_provide_context(self, user_input: str) -> bool:
        """
        Determine if the user input should trigger EchoZero context.

        Returns True if the input contains EchoZero-related keywords.
        """
        input_lower = user_input.lower()

        # Check for EchoZero-specific keywords
        has_echozero_keywords = any(keyword in input_lower for keyword in self.echozero_keywords)

        # Check for AI agent mentions (to avoid triggering on non-EchoZero AI chat)
        has_agent_mention = any(agent in input_lower for agent in self.agent_identifiers)

        # Check for development/code-related terms
        dev_terms = ['implement', 'create', 'add', 'new', 'code', 'function']
        has_dev_terms = any(term in input_lower for term in dev_terms)

        return has_echozero_keywords or (has_dev_terms and len(user_input.split()) < 20)

    def get_context_for_input(self, user_input: str, agent_id: str = "cursor_ai") -> Optional[Dict[str, Any]]:
        """
        Get EchoZero context for user input.

        Returns context dict if input should get context, None otherwise.
        """
        if not self.should_provide_context(user_input):
            return None

        # Check cache first
        cache_key = f"{agent_id}:{hash(user_input)}"
        if cache_key in self.context_cache:
            cached = self.context_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_timeout:
                return cached['context']

        # Get fresh context
        try:
            sys.path.insert(0, str(self.agent_assets_root / "scripts"))
            from context_provider import AgentAssetsContextProvider, ContextRequest

            provider = AgentAssetsContextProvider()
            request = ContextRequest(
                agent_id=agent_id,
                query=user_input,
                context_type="general"
            )

            context_response = provider.get_context_for_agent(request)

            # Format for AI agent consumption
            formatted_context = self._format_context_for_ai(context_response, user_input)

            # Cache the result
            self.context_cache[cache_key] = {
                'context': formatted_context,
                'timestamp': time.time()
            }

            return formatted_context

        except Exception as e:
            print(f"Context hook error: {e}")
            return None

    def _format_context_for_ai(self, context_response: Any, user_input: str) -> Dict[str, Any]:
        """Format context response for AI agent consumption."""
        try:
            core_context = context_response.core_context
            agent_context = context_response.agent_specific_context
            learning_insights = context_response.learning_insights

            # Create a clean, AI-friendly format
            formatted = {
                "echozero_context_provided": True,
                "user_request": user_input,
                "relevance_score": context_response.relevance_score,
                "core_guidance": {},
                "implementation_requirements": {},
                "quality_standards": {},
                "available_resources": {}
            }

            # Extract core values
            if "core_values" in core_context:
                formatted["core_guidance"]["values"] = core_context["core_values"]

            # Extract architectural patterns
            if "architectural_patterns" in core_context:
                formatted["core_guidance"]["patterns"] = core_context["architectural_patterns"]

            # Extract implementation requirements
            if "query_specific" in core_context:
                formatted["implementation_requirements"] = core_context["query_specific"]

            # Extract quality standards
            if "quality_guidelines" in core_context:
                formatted["quality_standards"] = core_context["quality_guidelines"]

            # Extract available resources
            if "current_capabilities" in core_context:
                formatted["available_resources"] = core_context["current_capabilities"]

            # Add agent-specific insights
            if agent_context:
                formatted["personal_insights"] = agent_context

            # Add learning insights
            if learning_insights:
                formatted["learning_context"] = learning_insights

            return formatted

        except Exception as e:
            return {
                "echozero_context_provided": True,
                "error": f"Formatting error: {e}",
                "basic_guidance": "Follow EchoZero BlockProcessor interface, add auto-registration, include error handling"
            }

    def inject_context_into_prompt(self, user_input: str, agent_id: str = "cursor_ai") -> str:
        """
        Inject EchoZero context into the user input for the AI agent.

        Returns the enhanced prompt with context.
        """
        context = self.get_context_for_input(user_input, agent_id)

        if not context:
            return user_input  # No context needed

        # Create enhanced prompt
        enhanced_prompt = f"""
## 🎯 EchoZero Development Context

**Your Task:** {user_input}

**EchoZero Context Provided:** ✅ (Relevance: {context.get('relevance_score', 0):.2f})

"""

        # Add core guidance
        if "core_guidance" in context and context["core_guidance"]:
            enhanced_prompt += "### Core EchoZero Principles\n"
            guidance = context["core_guidance"]

            if "values" in guidance:
                for value_name, value_info in guidance["values"].items():
                    if isinstance(value_info, dict) and "principle" in value_info:
                        enhanced_prompt += f"- **{value_name.replace('_', ' ').title()}:** {value_info['principle']}\n"

            if "patterns" in guidance:
                enhanced_prompt += "\n### Required Architectural Patterns\n"
                patterns = guidance["patterns"]
                if "key_patterns" in patterns:
                    for pattern, desc in patterns["key_patterns"].items():
                        enhanced_prompt += f"- **{pattern}:** {desc}\n"

        # Add implementation requirements
        if "implementation_requirements" in context and context["implementation_requirements"]:
            enhanced_prompt += "\n### Implementation Requirements\n"
            reqs = context["implementation_requirements"]
            for key, value in reqs.items():
                if isinstance(value, dict):
                    enhanced_prompt += f"**{key.replace('_', ' ').title()}:**\n"
                    for sub_key, sub_value in value.items():
                        enhanced_prompt += f"  - {sub_key.replace('_', ' ').title()}: {sub_value}\n"
                else:
                    enhanced_prompt += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        # Add quality standards
        if "quality_standards" in context and context["quality_standards"]:
            enhanced_prompt += "\n### Quality Standards (MANDATORY)\n"
            standards = context["quality_standards"]
            if "code_quality" in standards:
                for standard, desc in standards["code_quality"].items():
                    enhanced_prompt += f"- **{standard.replace('_', ' ').title()}:** {desc}\n"

        # Add available resources
        if "available_resources" in context and context["available_resources"]:
            enhanced_prompt += "\n### Available Resources\n"
            resources = context["available_resources"]
            if "available_blocks" in resources:
                enhanced_prompt += f"- **Existing Blocks:** {', '.join(resources['available_blocks'])}\n"
            if "supported_formats" in resources:
                enhanced_prompt += f"- **Supported Formats:** {', '.join(resources['supported_formats'])}\n"

        # Add personal insights
        if "personal_insights" in context and context["personal_insights"]:
            enhanced_prompt += "\n### Your Successful Patterns\n"
            insights = context["personal_insights"]
            if "successful_patterns" in insights and insights["successful_patterns"]:
                enhanced_prompt += f"- **Your proven approaches:** {', '.join(insights['successful_patterns'])}\n"

        # Add final instructions
        enhanced_prompt += """

### Implementation Instructions
1. **Follow interfaces exactly** - BlockProcessor, Command, etc.
2. **Add auto-registration** - register_processor_class() at end
3. **Include error handling** - ProcessingError with proper context
4. **Add comprehensive documentation** - docstrings, type hints
5. **Create tests** - corresponding test files required
6. **Question additions** - prefer existing patterns

**Remember:** "Best part is no part" - prefer reuse over creation!

---
**Now implement the requested feature following these EchoZero patterns.**
---

"""

        return enhanced_prompt + user_input

    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_items": len(self.context_cache),
            "cache_size_mb": sum(len(json.dumps(item)) for item in self.context_cache.values()) / (1024 * 1024)
        }


# Global hook instance
_ai_context_hook = AICursorContextHook()


def get_ai_context(user_input: str, agent_id: str = "cursor_ai") -> Optional[Dict[str, Any]]:
    """Get AI context for user input."""
    return _ai_context_hook.get_context_for_input(user_input, agent_id)


def inject_context_into_prompt(user_input: str, agent_id: str = "cursor_ai") -> str:
    """Inject context into AI prompt."""
    return _ai_context_hook.inject_context_into_prompt(user_input, agent_id)


def clear_ai_context_cache():
    """Clear the AI context cache."""
    _ai_context_hook.clear_cache()


def get_ai_context_stats() -> Dict[str, Any]:
    """Get AI context cache statistics."""
    return _ai_context_hook.get_cache_stats()


# For testing and direct usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_context_hook.py \"Your request here\" [agent_id]")
        print("Example: python ai_context_hook.py \"Make a new block\" claude-3")
        sys.exit(1)

    user_input = sys.argv[1]
    agent_id = sys.argv[2] if len(sys.argv) > 2 else "cursor_ai"

    # Test context injection
    enhanced_prompt = inject_context_into_prompt(user_input, agent_id)

    print("=== ENHANCED PROMPT FOR AI AGENT ===")
    print(enhanced_prompt)
    print("\n=== CACHE STATS ===")
    print(json.dumps(get_ai_context_stats(), indent=2))

