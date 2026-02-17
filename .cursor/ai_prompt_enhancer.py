#!/usr/bin/env python3
"""
AI Prompt Enhancer for Cursor IDE

Automatically enhances AI prompts with EchoZero context before they are sent to AI agents.
This provides seamless integration where users just type natural language and get enhanced prompts.
"""

import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional


class AIPromptEnhancer:
    """Enhances AI prompts with EchoZero context automatically."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.agent_assets_root = self.project_root / "AgentAssets"
        self.enhancement_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 600  # 10 minutes

        # Keywords that trigger EchoZero enhancement
        self.echozero_triggers = {
            # Block-related
            'block', 'processor', 'audio', 'processing', 'analyze', 'transform',

            # UI-related
            'ui', 'interface', 'widget', 'panel', 'component', 'display',

            # Command-related
            'command', 'action', 'undo', 'redo', 'execute',

            # Testing
            'test', 'testing', 'pytest', 'unit test',

            # General development
            'implement', 'create', 'add', 'new', 'functionality', 'feature',
            'code', 'develop', 'build', 'make',

            # EchoZero specific
            'echozero', 'echozero', 'audio processing', 'music', 'sound'
        }

    def should_enhance_prompt(self, prompt: str) -> bool:
        """
        Determine if a prompt should be enhanced with EchoZero context.

        Returns True if the prompt contains EchoZero-related keywords.
        """
        prompt_lower = prompt.lower()

        # Check for EchoZero trigger words
        has_triggers = any(trigger in prompt_lower for trigger in self.echozero_triggers)

        # Additional checks for development context
        has_dev_context = (
            ('implement' in prompt_lower or 'create' in prompt_lower) and
            len(prompt.split()) < 50  # Keep it to reasonable length prompts
        )

        return has_triggers or has_dev_context

    def enhance_prompt(self, original_prompt: str, agent_id: str = "cursor_ai") -> str:
        """
        Enhance a prompt with EchoZero context.

        Returns the enhanced prompt if enhancement is needed, otherwise returns original.
        """
        if not self.should_enhance_prompt(original_prompt):
            return original_prompt

        # Check cache first
        cache_key = f"{agent_id}:{hash(original_prompt)}"
        if cache_key in self.enhancement_cache:
            cached = self.enhancement_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_timeout:
                return cached['enhanced_prompt']

        try:
            # Get context using the AI context hook
            sys.path.insert(0, str(self.project_root / ".cursor"))
            from ai_context_hook import inject_context_into_prompt

            enhanced_prompt = inject_context_into_prompt(original_prompt, agent_id)

            # Cache the result
            self.enhancement_cache[cache_key] = {
                'enhanced_prompt': enhanced_prompt,
                'timestamp': time.time(),
                'original_prompt': original_prompt
            }

            return enhanced_prompt

        except Exception as e:
            print(f"Prompt enhancement error: {e}")
            # Return original prompt if enhancement fails
            return original_prompt

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt enhancements."""
        return {
            "cache_size": len(self.enhancement_cache),
            "cache_entries": list(self.enhancement_cache.keys())[:5],  # First 5 keys
            "triggers": list(self.echozero_triggers)
        }

    def clear_cache(self):
        """Clear the enhancement cache."""
        self.enhancement_cache.clear()


# Global enhancer instance
_prompt_enhancer = AIPromptEnhancer()


def enhance_ai_prompt(prompt: str, agent_id: str = "cursor_ai") -> str:
    """Enhance an AI prompt with EchoZero context."""
    return _prompt_enhancer.enhance_prompt(prompt, agent_id)


def should_enhance_prompt(prompt: str) -> bool:
    """Check if a prompt should be enhanced."""
    return _prompt_enhancer.should_enhance_prompt(prompt)


def get_enhancement_stats() -> Dict[str, Any]:
    """Get enhancement statistics."""
    return _prompt_enhancer.get_enhancement_stats()


def clear_enhancement_cache():
    """Clear the enhancement cache."""
    _prompt_enhancer.clear_cache()


# For testing and direct usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_prompt_enhancer.py \"Your prompt here\" [agent_id]")
        print("Example: python ai_prompt_enhancer.py \"Make a new block\" claude-3")
        sys.exit(1)

    user_prompt = sys.argv[1]
    agent_id = sys.argv[2] if len(sys.argv) > 2 else "cursor_ai"

    print("Original prompt:")
    print(f'"{user_prompt}"')
    print()

    if should_enhance_prompt(user_prompt):
        enhanced = enhance_ai_prompt(user_prompt, agent_id)
        print("Enhanced prompt:")
        print("=" * 50)
        print(enhanced)
        print("=" * 50)
    else:
        print("Prompt does not need enhancement (no EchoZero keywords detected)")

    print("\nEnhancement stats:")
    print(json.dumps(get_enhancement_stats(), indent=2))

