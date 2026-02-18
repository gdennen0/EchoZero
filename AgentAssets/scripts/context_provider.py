#!/usr/bin/env python3
"""
AgentAssets Context Provider for AI Agents.

Provides intelligent, learning-based context to AI agents through chat interfaces.
Core learning applies to all agents, while specific learning adapts to individual agents.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class AgentProfile:
    """Profile for individual AI agents."""
    agent_id: str
    model_name: str
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_profile: Dict[str, Any] = field(default_factory=dict)
    context_preferences: Dict[str, float] = field(default_factory=dict)
    success_patterns: List[str] = field(default_factory=list)
    last_interaction: Optional[datetime] = None
    total_interactions: int = 0
    effectiveness_score: float = 0.5


@dataclass
class ContextRequest:
    """Request for context from an AI agent."""
    agent_id: str
    query: str
    context_type: str = "general"  # general, specific, task, domain
    domain: Optional[str] = None
    current_task: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResponse:
    """Response with contextual information."""
    request_id: str
    agent_id: str
    core_context: Dict[str, Any]
    agent_specific_context: Dict[str, Any]
    learning_insights: Dict[str, Any]
    relevance_score: float
    generated_at: datetime
    cache_key: str


class AgentAssetsContextProvider:
    """Provides intelligent context to AI agents based on learned patterns."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.agent_assets_root = Path(__file__).resolve().parent.parent

        # Import learning engine
        sys.path.insert(0, str(self.agent_assets_root / "scripts"))
        try:
            from learning_engine import SelfRefiningLearningEngine
            self.learning_engine = SelfRefiningLearningEngine()
        except ImportError:
            print("Warning: Learning engine not available")
            self.learning_engine = None

        # Agent profiles and data
        self.data_dir = self.agent_assets_root / "data"
        self.profiles_file = self.data_dir / "agent_profiles.json"
        self.context_cache_file = self.data_dir / "context_cache.json"

        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.context_cache: Dict[str, ContextResponse] = {}
        self.cache_timeout = 1800  # 30 minutes

        # Load existing data
        self._load_agent_profiles()
        self._load_context_cache()

    def get_context_for_agent(self, request: ContextRequest) -> ContextResponse:
        """
        Get intelligent context for an AI agent based on their request and learning history.

        This is the main entry point for AI agents to request context.
        """
        request_id = f"ctx_{int(datetime.now().timestamp())}_{request.agent_id[:8]}"

        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_cached_response(cache_key, request)
        if cached_response:
            # Record usage for learning
            self._record_context_usage(request, cached_response)
            return cached_response

        # Get agent profile (create if doesn't exist)
        profile = self._get_or_create_agent_profile(request.agent_id)

        # Generate core context (shared across all agents)
        core_context = self._generate_core_context(request)

        # Generate agent-specific context
        agent_context = self._generate_agent_specific_context(request, profile)

        # Generate learning insights
        learning_insights = self._generate_learning_insights(request, profile)

        # Calculate overall relevance
        relevance_score = self._calculate_relevance_score(core_context, agent_context, learning_insights)

        # Create response
        response = ContextResponse(
            request_id=request_id,
            agent_id=request.agent_id,
            core_context=core_context,
            agent_specific_context=agent_context,
            learning_insights=learning_insights,
            relevance_score=relevance_score,
            generated_at=datetime.now(),
            cache_key=cache_key
        )

        # Cache the response
        self._cache_response(cache_key, response)

        # Update agent profile and learning
        self._update_agent_profile(request, response, profile)
        self._record_context_usage(request, response)

        return response

    def get_agent_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get insights about a specific agent's learning and preferences."""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return {"error": "Agent profile not found"}

        # Get learning statistics
        learning_stats = self._calculate_agent_learning_stats(profile)

        # Get context usage patterns
        usage_patterns = self._analyze_context_usage_patterns(agent_id)

        # Get effectiveness trends
        effectiveness_trends = self._calculate_effectiveness_trends(profile)

        return {
            "agent_id": agent_id,
            "profile": {
                "model_name": profile.model_name,
                "total_interactions": profile.total_interactions,
                "last_interaction": profile.last_interaction.isoformat() if profile.last_interaction else None,
                "effectiveness_score": profile.effectiveness_score
            },
            "learning_stats": learning_stats,
            "usage_patterns": usage_patterns,
            "effectiveness_trends": effectiveness_trends,
            "recommendations": self._generate_agent_recommendations(profile)
        }

    def update_agent_feedback(self, agent_id: str, feedback: Dict[str, Any]) -> bool:
        """Update agent profile based on feedback about provided context."""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return False

        # Update effectiveness score based on feedback
        if "context_helpful" in feedback:
            helpfulness = feedback["context_helpful"]  # 0.0 to 1.0
            # Weighted moving average
            profile.effectiveness_score = (profile.effectiveness_score * 0.7) + (helpfulness * 0.3)

        # Update preferences based on feedback
        if "preferred_context_types" in feedback:
            for ctx_type, preference in feedback["preferred_context_types"].items():
                profile.context_preferences[ctx_type] = preference

        # Record successful patterns
        if "successful_patterns" in feedback:
            for pattern in feedback["successful_patterns"]:
                if pattern not in profile.success_patterns:
                    profile.success_patterns.append(pattern)

        # Save updated profile
        self._save_agent_profiles()

        return True

    def _generate_core_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Generate core context shared across all agents."""
        core_context = {
            "core_values": self._get_core_values_context(),
            "architectural_patterns": self._get_architectural_context(request),
            "domain_knowledge": self._get_domain_context(request),
            "quality_guidelines": self._get_quality_context(),
            "current_capabilities": self._get_capabilities_context()
        }

        # Add learning-engine-based context if available
        if self.learning_engine:
            try:
                # Get core understanding from learning engine
                core_understanding = self.learning_engine.core_understanding

                # Add core domain concepts
                if hasattr(core_understanding, 'domain_concepts') and core_understanding.domain_concepts:
                    core_context["learned_domain_concepts"] = dict(list(core_understanding.domain_concepts.items())[:3])

                # Add core architectural patterns
                if hasattr(core_understanding, 'architectural_patterns') and core_understanding.architectural_patterns:
                    core_context["learned_architectural_patterns"] = core_understanding.architectural_patterns

                # Add evolution insights
                if hasattr(core_understanding, 'evolution_timeline') and core_understanding.evolution_timeline:
                    latest_evolution = core_understanding.evolution_timeline[-1] if core_understanding.evolution_timeline else {}
                    core_context["system_evolution"] = {
                        "latest_update": latest_evolution.get("timestamp", ""),
                        "improvements": latest_evolution.get("quality_improvements", {})
                    }
            except Exception as e:
                # If learning engine has issues, provide basic context
                core_context["learning_status"] = "basic_mode"

        # Add query-specific context if provided
        if request.query:
            core_context["query_specific"] = self._analyze_query_for_context(request.query)

        return core_context

    def _generate_agent_specific_context(self, request: ContextRequest, profile: AgentProfile) -> Dict[str, Any]:
        """Generate context specific to this agent's learning history."""
        specific_context = {}

        # Include successful patterns from this agent
        if profile.success_patterns:
            specific_context["successful_patterns"] = profile.success_patterns[:5]  # Top 5

        # Include preferred context types
        if profile.context_preferences:
            specific_context["preferred_types"] = profile.context_preferences

        # Include learning profile insights
        if profile.learning_profile:
            specific_context["learning_insights"] = profile.learning_profile

        # Add interaction history patterns
        if len(profile.interaction_history) >= 3:
            specific_context["interaction_patterns"] = self._analyze_interaction_patterns(profile)

        # Add task-specific context based on current task
        if request.current_task:
            specific_context["task_context"] = self._get_task_specific_context(request.current_task, profile)

        # Add agent-specific learning from learning engine
        if self.learning_engine and request.agent_id in self.learning_engine.agent_understandings:
            agent_understanding = self.learning_engine.agent_understandings[request.agent_id]

            # Add preferred patterns
            if agent_understanding.preferred_patterns:
                specific_context["preferred_patterns"] = dict(list(agent_understanding.preferred_patterns.items())[:5])

            # Add successful approaches
            if agent_understanding.successful_approaches:
                specific_context["successful_approaches"] = agent_understanding.successful_approaches[:3]

            # Add adaptation patterns
            if agent_understanding.adaptation_patterns:
                specific_context["adaptation_patterns"] = agent_understanding.adaptation_patterns

        return specific_context

    def _generate_learning_insights(self, request: ContextRequest, profile: AgentProfile) -> Dict[str, Any]:
        """Generate insights based on learning patterns."""
        insights = {}

        # Get recent learning from the system
        if self.learning_engine:
            try:
                # Get insights about the current query/topic
                if request.query:
                    query_insights = self.learning_engine.get_context_for_query(request.query)
                    insights["query_insights"] = {
                        "confidence_score": query_insights.get("confidence_score", 0),
                        "key_patterns": [p.get("pattern_id", "") for p in query_insights.get("relevant_patterns", [])][:3]
                    }

                # Get evolution insights
                system_evolution = self.learning_engine.understanding.get("evolution_timeline", [])
                if system_evolution:
                    latest_evolution = system_evolution[-1]
                    insights["system_evolution"] = {
                        "latest_update": latest_evolution.get("timestamp", ""),
                        "improvements": latest_evolution.get("quality_improvements", {})
                    }

            except Exception as e:
                insights["error"] = f"Learning engine unavailable: {e}"

        # Add agent-specific learning insights
        insights["agent_learning"] = {
            "effectiveness_trend": self._calculate_learning_trend(profile),
            "preferred_domains": list(profile.learning_profile.get("preferred_domains", [])),
            "improvement_areas": profile.learning_profile.get("improvement_areas", [])
        }

        return insights

    def _get_core_values_context(self) -> Dict[str, Any]:
        """Get core values context."""
        return {
            "best_part_is_no_part": {
                "principle": "Question every addition, prefer removal",
                "application": "Before adding code, ask 'Can we delete instead?'",
                "examples": [
                    "Use existing patterns rather than creating new abstractions",
                    "Remove unused code and dependencies",
                    "Prefer simple solutions over complex ones"
                ]
            },
            "simplicity_and_refinement": {
                "principle": "Simple is what remains after removing unnecessary parts",
                "application": "Refine existing features before adding new ones",
                "examples": [
                    "Polish existing functionality rather than adding scope",
                    "Make common cases easy, complex cases possible",
                    "Explicit over implicit design"
                ]
            }
        }

    def _get_architectural_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Get architectural context."""
        context = {
            "layered_architecture": {
                "layers": ["Presentation", "Application", "Domain", "Infrastructure"],
                "principles": ["Clear separation of concerns", "Dependency inversion", "Single responsibility"]
            },
            "key_patterns": {
                "facade_pattern": "ApplicationFacade provides unified API",
                "repository_pattern": "Domain defines interfaces, infrastructure implements",
                "command_pattern": "All undoable operations use CommandBus",
                "strategy_pattern": "Block processors implement common interface"
            }
        }

        # Add domain-specific architectural context
        if request.domain:
            context["domain_specific"] = self._get_domain_architectural_context(request.domain)

        return context

    def _get_domain_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Get domain-specific context."""
        # This would pull from the learning engine's understanding
        domain_context = {}

        if self.learning_engine and hasattr(self.learning_engine, 'understanding'):
            domain_knowledge = self.learning_engine.understanding.get("domain_concepts", {})

            if request.domain and request.domain in domain_knowledge:
                domain_context[request.domain] = domain_knowledge[request.domain]
            else:
                # Provide general domain guidance
                domain_context["general"] = {
                    "audio_processing": "Use librosa for loading, focus on efficient memory usage",
                    "block_system": "Implement BlockProcessor interface, auto-register processors",
                    "command_system": "Extend EchoZeroCommand, use CommandBus.execute()",
                    "ui_components": "Follow existing design patterns, use ApplicationFacade"
                }

        return domain_context

    def _get_quality_context(self) -> Dict[str, Any]:
        """Get quality guidelines context."""
        return {
            "code_quality": {
                "docstrings": "All classes and functions need docstrings",
                "error_handling": "Explicit error handling with CommandResult",
                "testing": "Unit tests for all new functionality",
                "type_hints": "Use type hints for better code clarity"
            },
            "architectural_quality": {
                "layer_separation": "UI never accesses domain/infrastructure directly",
                "dependency_injection": "Use constructor injection for dependencies",
                "single_responsibility": "Each class/function has one clear purpose",
                "interface_consistency": "Follow established naming and structure patterns"
            }
        }

    def _get_capabilities_context(self) -> Dict[str, Any]:
        """Get current system capabilities."""
        # This would be pulled from CURRENT_STATE.md or the learning engine
        return {
            "available_blocks": [
                "LoadAudio", "DetectOnsets", "SeparatorBlock", "Editor", "ExportAudio"
            ],
            "supported_formats": ["WAV", "MP3", "audio processing with ML"],
            "key_features": [
                "Real-time audio processing", "Block-based workflow", "Qt GUI", "CLI interface",
                "Project persistence", "Undo/redo system", "Comprehensive testing"
            ],
            "current_limitations": [
                "Sequential execution only", "No real-time audio during editing"
            ]
        }

    def _analyze_query_for_context(self, query: str) -> Dict[str, Any]:
        """Analyze a query to provide relevant context."""
        query_lower = query.lower()
        context = {}

        # Detect intent and provide relevant context
        if any(word in query_lower for word in ["block", "processor", "create", "implement"]):
            context["intent"] = "block_creation"
            context["relevant_info"] = {
                "interface": "Implement BlockProcessor with can_process() and process()",
                "registration": "Auto-registration happens in module __init__.py",
                "patterns": "Follow existing block patterns for consistency",
                "testing": "Create corresponding test files"
            }

        elif any(word in query_lower for word in ["command", "undo", "redo"]):
            context["intent"] = "command_creation"
            context["relevant_info"] = {
                "base_class": "Extend EchoZeroCommand",
                "methods": "Implement redo() and undo() methods",
                "execution": "Use CommandBus.execute() to run commands",
                "result": "Return CommandResult objects"
            }

        elif any(word in query_lower for word in ["ui", "gui", "interface", "widget"]):
            context["intent"] = "ui_creation"
            context["relevant_info"] = {
                "facade": "Use ApplicationFacade for all UI operations",
                "patterns": "Follow existing UI component patterns",
                "separation": "Keep UI separate from business logic",
                "testing": "Test UI components separately"
            }

        elif any(word in query_lower for word in ["test", "testing", "pytest"]):
            context["intent"] = "testing"
            context["relevant_info"] = {
                "framework": "Use pytest for all testing",
                "structure": "Follow existing test file naming and structure",
                "coverage": "Aim for comprehensive test coverage",
                "fixtures": "Use pytest fixtures for test setup"
            }

        return context

    def _get_or_create_agent_profile(self, agent_id: str) -> AgentProfile:
        """Get existing agent profile or create new one."""
        if agent_id not in self.agent_profiles:
            # Try to infer model from agent_id
            model_name = self._infer_model_from_id(agent_id)

            self.agent_profiles[agent_id] = AgentProfile(
                agent_id=agent_id,
                model_name=model_name
            )

            # Save immediately
            self._save_agent_profiles()

        return self.agent_profiles[agent_id]

    def _infer_model_from_id(self, agent_id: str) -> str:
        """Infer model name from agent ID."""
        agent_lower = agent_id.lower()

        if "gpt" in agent_lower or "chatgpt" in agent_lower:
            return "GPT"
        elif "claude" in agent_lower:
            return "Claude"
        elif "gemini" in agent_lower or "bard" in agent_lower:
            return "Gemini"
        elif "groq" in agent_lower:
            return "Groq"
        else:
            return "Unknown"

    def _update_agent_profile(self, request: ContextRequest, response: ContextResponse, profile: AgentProfile):
        """Update agent profile based on interaction."""
        # Update interaction history
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "query": request.query,
            "context_type": request.context_type,
            "relevance_score": response.relevance_score,
            "context_used": {
                "core_context_keys": list(response.core_context.keys()),
                "agent_context_keys": list(response.agent_specific_context.keys())
            }
        }

        profile.interaction_history.append(interaction_record)
        profile.last_interaction = datetime.now()
        profile.total_interactions += 1

        # Keep only last 50 interactions
        if len(profile.interaction_history) > 50:
            profile.interaction_history = profile.interaction_history[-50:]

        # Update learning profile
        self._update_learning_profile(profile, request, response)

        # Save profiles
        self._save_agent_profiles()

    def _update_learning_profile(self, profile: AgentProfile, request: ContextRequest, response: ContextResponse):
        """Update the learning profile for an agent."""
        # Track preferred context types
        context_types_used = []
        if response.core_context:
            context_types_used.extend(response.core_context.keys())
        if response.agent_specific_context:
            context_types_used.extend(response.agent_specific_context.keys())

        for ctx_type in context_types_used:
            if ctx_type not in profile.context_preferences:
                profile.context_preferences[ctx_type] = 0.5  # Default preference

        # Track domains of interest
        if request.domain:
            if "preferred_domains" not in profile.learning_profile:
                profile.learning_profile["preferred_domains"] = {}
            profile.learning_profile["preferred_domains"][request.domain] = \
                profile.learning_profile["preferred_domains"].get(request.domain, 0) + 1

        # Track improvement areas based on relevance scores
        if response.relevance_score < 0.7:
            if "improvement_areas" not in profile.learning_profile:
                profile.learning_profile["improvement_areas"] = []
            if "context_relevance" not in profile.learning_profile["improvement_areas"]:
                profile.learning_profile["improvement_areas"].append("context_relevance")

    def _record_context_usage(self, request: ContextRequest, response: ContextResponse):
        """Record context usage for learning engine."""
        if self.learning_engine:
            usage_data = {
                "context_type": "agent_context_request",
                "agent_id": request.agent_id,
                "query": request.query,
                "context_provided": {
                    "core_context": bool(response.core_context),
                    "agent_specific": bool(response.agent_specific_context),
                    "learning_insights": bool(response.learning_insights)
                },
                "relevance_score": response.relevance_score
            }

            try:
                self.learning_engine.record_usage_event("context_provider", usage_data)
            except:
                pass  # Don't fail if learning engine is unavailable

    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate cache key for request."""
        key_components = [
            request.agent_id,
            request.query,
            request.context_type,
            str(request.domain),
            str(request.current_task)
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str, request: ContextRequest) -> Optional[ContextResponse]:
        """Get cached response if valid."""
        if cache_key not in self.context_cache:
            return None

        cached = self.context_cache[cache_key]
        cache_age = datetime.now() - cached.generated_at

        # Check if cache is still valid (within timeout and agent hasn't changed significantly)
        if cache_age.total_seconds() < self.cache_timeout:
            return cached

        # Remove expired cache
        del self.context_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: ContextResponse):
        """Cache a response."""
        self.context_cache[cache_key] = response

        # Limit cache size
        if len(self.context_cache) > 50:
            # Remove oldest entries
            sorted_cache = sorted(self.context_cache.items(),
                                key=lambda x: x[1].generated_at)
            to_remove = sorted_cache[:10]
            for key, _ in to_remove:
                del self.context_cache[key]

    def _calculate_relevance_score(self, core_context: Dict, agent_context: Dict, learning_insights: Dict) -> float:
        """Calculate overall relevance score for context."""
        scores = []

        # Core context score
        if core_context:
            core_score = min(len(core_context) / 5, 1.0)  # Up to 5 core context items
            scores.append(core_score)

        # Agent context score
        if agent_context:
            agent_score = min(len(agent_context) / 3, 1.0)  # Up to 3 agent context items
            scores.append(agent_score * 0.8)  # Slightly lower weight

        # Learning insights score
        if learning_insights:
            learning_score = min(len(learning_insights) / 2, 1.0)  # Up to 2 insight categories
            scores.append(learning_score * 0.6)  # Lower weight for insights

        return sum(scores) / len(scores) if scores else 0.0

    def _load_agent_profiles(self):
        """Load agent profiles from disk."""
        if not self.profiles_file.exists():
            return

        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                for agent_id, profile_data in data.items():
                    # Convert timestamp strings back to datetime
                    if 'last_interaction' in profile_data and profile_data['last_interaction']:
                        profile_data['last_interaction'] = datetime.fromisoformat(profile_data['last_interaction'])

                    profile = AgentProfile(**profile_data)
                    self.agent_profiles[agent_id] = profile
        except:
            pass

    def _save_agent_profiles(self):
        """Save agent profiles to disk."""
        data = {}
        for agent_id, profile in self.agent_profiles.items():
            profile_data = {
                **profile.__dict__,
                'last_interaction': profile.last_interaction.isoformat() if profile.last_interaction else None,
                'interaction_history': profile.interaction_history[-10:]  # Keep last 10 interactions
            }
            data[agent_id] = profile_data

        with open(self.profiles_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_context_cache(self):
        """Load context cache from disk."""
        if not self.context_cache_file.exists():
            return

        try:
            with open(self.context_cache_file, 'r') as f:
                data = json.load(f)
                for cache_key, response_data in data.items():
                    # Convert timestamp string back to datetime
                    if 'generated_at' in response_data:
                        response_data['generated_at'] = datetime.fromisoformat(response_data['generated_at'])

                    response = ContextResponse(**response_data)
                    self.context_cache[cache_key] = response
        except:
            pass

    def _save_context_cache(self):
        """Save context cache to disk."""
        data = {}
        for cache_key, response in self.context_cache.items():
            response_data = {
                **response.__dict__,
                'generated_at': response.generated_at.isoformat()
            }
            data[cache_key] = response_data

        with open(self.context_cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    # Helper methods for analysis
    def _calculate_agent_learning_stats(self, profile: AgentProfile) -> Dict[str, Any]:
        return {
            "total_interactions": profile.total_interactions,
            "avg_relevance_score": profile.effectiveness_score,
            "preferred_context_types": profile.context_preferences,
            "learning_progress": len(profile.success_patterns)
        }

    def _analyze_context_usage_patterns(self, agent_id: str) -> Dict[str, Any]:
        return {"patterns": "analysis_not_implemented"}

    def _calculate_effectiveness_trends(self, profile: AgentProfile) -> Dict[str, Any]:
        return {"trends": "analysis_not_implemented"}

    def _generate_agent_recommendations(self, profile: AgentProfile) -> List[str]:
        return ["Continue using the context system"]

    def _calculate_learning_trend(self, profile: AgentProfile) -> str:
        return "improving"

    def _analyze_interaction_patterns(self, profile: AgentProfile) -> Dict[str, Any]:
        return {"patterns": []}

    def _get_task_specific_context(self, task: str, profile: AgentProfile) -> Dict[str, Any]:
        return {"task": task}

    def _get_domain_architectural_context(self, domain: str) -> Dict[str, Any]:
        return {"domain": domain}


def main():
    """CLI interface for context provider."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python context_provider.py request <agent_id> <query>")
        print("  python context_provider.py insights <agent_id>")
        print("  python context_provider.py feedback <agent_id> <helpful_score>")
        sys.exit(1)

    provider = AgentAssetsContextProvider()
    command = sys.argv[1]

    if command == "request" and len(sys.argv) >= 4:
        agent_id = sys.argv[2]
        query = " ".join(sys.argv[3:])

        request = ContextRequest(
            agent_id=agent_id,
            query=query
        )

        response = provider.get_context_for_agent(request)

        print("Context Response:")
        print(f"Request ID: {response.request_id}")
        print(f"Relevance Score: {response.relevance_score:.2f}")
        print("\nCore Context:")
        for key, value in response.core_context.items():
            print(f"  {key}: {str(value)[:100]}...")
        print("\nAgent-Specific Context:")
        for key, value in response.agent_specific_context.items():
            print(f"  {key}: {str(value)[:100]}...")
        print("\nLearning Insights:")
        for key, value in response.learning_insights.items():
            print(f"  {key}: {str(value)[:100]}...")

    elif command == "insights" and len(sys.argv) >= 3:
        agent_id = sys.argv[2]
        insights = provider.get_agent_insights(agent_id)
        print(json.dumps(insights, indent=2))

    elif command == "feedback" and len(sys.argv) >= 4:
        agent_id = sys.argv[2]
        helpful_score = float(sys.argv[3])

        feedback = {"context_helpful": helpful_score}
        success = provider.update_agent_feedback(agent_id, feedback)

        print(f"Feedback {'recorded' if success else 'failed'} for {agent_id}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
