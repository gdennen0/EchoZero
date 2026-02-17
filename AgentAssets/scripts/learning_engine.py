#!/usr/bin/env python3
"""
Self-Refining Learning Engine for AgentAssets.

This system continuously learns from AI agent interactions, refines understanding
of the codebase, and improves the quality and relevance of provided information.
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import ast
import subprocess
import sys


@dataclass
class CodePattern:
    """Represents a learned code pattern."""
    pattern_id: str
    pattern_type: str  # 'class', 'function', 'block', 'command', etc.
    signature: str
    context: Dict[str, Any]
    usage_count: int = 0
    success_rate: float = 0.0
    last_seen: datetime = None
    quality_score: float = 0.5
    related_patterns: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreUnderstanding:
    """Core understanding shared across all agents."""
    domain_concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    architectural_patterns: Dict[str, Any] = field(default_factory=dict)
    quality_patterns: Dict[str, Any] = field(default_factory=dict)
    evolution_timeline: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    universal_patterns: Dict[str, CodePattern] = field(default_factory=dict)


@dataclass
class AgentSpecificUnderstanding:
    """Agent-specific understanding and preferences."""
    agent_id: str
    preferred_patterns: Dict[str, float] = field(default_factory=dict)  # pattern_id -> preference score
    successful_approaches: List[str] = field(default_factory=dict)
    learning_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_patterns: Dict[str, Any] = field(default_factory=dict)
    interaction_style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEvent:
    """Records a learning event for pattern analysis."""
    event_type: str  # 'contribution', 'validation', 'usage', 'feedback'
    agent_id: str
    timestamp: datetime
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    patterns_learned: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class SelfRefiningLearningEngine:
    """Engine that continuously learns and refines understanding."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.agent_assets_root = Path(__file__).parent.parent
        self.data_dir = self.agent_assets_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data files - separated for core vs agent-specific
        self.core_patterns_file = self.data_dir / "core_patterns.json"
        self.core_understanding_file = self.data_dir / "core_understanding.json"
        self.agent_understandings_file = self.data_dir / "agent_understandings.json"
        self.learning_events_file = self.data_dir / "learning_events.json"

        # Learning state - separated into core and agent-specific
        self.core_patterns: Dict[str, CodePattern] = {}  # Universal patterns
        self.core_understanding = CoreUnderstanding()
        self.agent_understandings: Dict[str, AgentSpecificUnderstanding] = {}  # Agent-specific
        self.learning_events: List[LearningEvent] = []

        # Load existing data
        self._load_state()

    def record_contribution_event(self, agent_id: str, contribution_data: Dict[str, Any]) -> str:
        """Record and learn from a contribution event."""
        event_id = f"contrib_{int(time.time())}_{agent_id}"

        # Analyze the contribution
        patterns_learned = self._analyze_contribution(contribution_data)
        quality_metrics = self._assess_contribution_quality(contribution_data)

        # Create learning event
        event = LearningEvent(
            event_type="contribution",
            agent_id=agent_id,
            timestamp=datetime.now(),
            context=contribution_data,
            outcome={"event_id": event_id},
            patterns_learned=patterns_learned,
            quality_metrics=quality_metrics
        )

        # Update patterns based on learning
        self._update_patterns_from_event(event)

        # Refine understanding
        self._refine_understanding(event)

        # Save the event
        self.learning_events.append(event)
        self._save_learning_events()

        return event_id

    def record_validation_event(self, agent_id: str, validation_data: Dict[str, Any]) -> str:
        """Record and learn from a validation event."""
        event_id = f"valid_{int(time.time())}_{agent_id}"

        # Analyze validation results
        insights = self._analyze_validation_results(validation_data)
        quality_metrics = self._extract_validation_metrics(validation_data)

        event = LearningEvent(
            event_type="validation",
            agent_id=agent_id,
            timestamp=datetime.now(),
            context=validation_data,
            outcome={"insights": insights, "event_id": event_id},
            patterns_learned=[],  # Validation events don't directly learn patterns
            quality_metrics=quality_metrics
        )

        # Learn from validation feedback
        self._learn_from_validation(event)

        self.learning_events.append(event)
        self._save_learning_events()

        return event_id

    def record_usage_event(self, agent_id: str, usage_data: Dict[str, Any]) -> str:
        """Record and learn from a usage event (when AI accesses information)."""
        event_id = f"usage_{int(time.time())}_{agent_id}"

        # Analyze what information was accessed and why
        access_patterns = self._analyze_information_access(usage_data)

        event = LearningEvent(
            event_type="usage",
            agent_id=agent_id,
            timestamp=datetime.now(),
            context=usage_data,
            outcome={"access_patterns": access_patterns, "event_id": event_id},
            patterns_learned=[],
            quality_metrics={}
        )

        # Learn from usage patterns
        self._learn_from_usage(event)

        self.learning_events.append(event)
        self._save_learning_events()

        return event_id

    def get_context_for_file(self, file_path: str, cursor_position: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Get relevant context for a specific file and cursor position."""
        file_path = Path(file_path)

        # Analyze file content
        file_analysis = self._analyze_file_content(file_path)

        # Get relevant patterns
        relevant_patterns = self._find_relevant_patterns(file_analysis)

        # Get understanding context
        understanding_context = self._get_understanding_context(file_analysis)

        # Get quality insights
        quality_insights = self._get_quality_insights(file_analysis)

        # Get evolution context
        evolution_context = self._get_evolution_context(file_analysis)

        context = {
            "file_analysis": file_analysis,
            "relevant_patterns": relevant_patterns,
            "understanding_context": understanding_context,
            "quality_insights": quality_insights,
            "evolution_context": evolution_context,
            "cursor_context": self._get_cursor_specific_context(file_path, cursor_position) if cursor_position else {},
            "confidence_score": self._calculate_context_confidence(relevant_patterns, understanding_context),
            "generated_at": datetime.now().isoformat()
        }

        # Record this usage for learning
        self.record_usage_event("system", {
            "context_type": "file_context",
            "file_path": str(file_path),
            "patterns_provided": [p.pattern_id for p in relevant_patterns],
            "understanding_keys": list(understanding_context.keys())
        })

        return context

    def get_context_for_query(self, query: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get context for a specific query or task."""
        # Analyze the query
        query_analysis = self._analyze_query(query)

        # Find relevant patterns
        relevant_patterns = self._find_patterns_for_query(query_analysis)

        # Get understanding context
        understanding_context = self._get_understanding_for_query(query_analysis)

        # Get historical context
        historical_context = self._get_historical_context(query_analysis)

        context = {
            "query_analysis": query_analysis,
            "relevant_patterns": relevant_patterns,
            "understanding_context": understanding_context,
            "historical_context": historical_context,
            "context_data": context_data or {},
            "confidence_score": self._calculate_query_confidence(query_analysis, relevant_patterns),
            "generated_at": datetime.now().isoformat()
        }

        # Record usage
        self.record_usage_event("system", {
            "context_type": "query_context",
            "query": query,
            "patterns_provided": [p.pattern_id for p in relevant_patterns],
            "understanding_keys": list(understanding_context.keys())
        })

        return context

    def refine_understanding(self) -> Dict[str, Any]:
        """Perform periodic refinement of the system's understanding."""
        print("🔄 Refining system understanding...")

        # Analyze recent events
        recent_events = [e for e in self.learning_events
                        if e.timestamp > datetime.now() - timedelta(days=7)]

        # Update pattern quality scores
        self._update_pattern_quality_scores(recent_events)

        # Refine architectural understanding
        self._refine_architectural_understanding(recent_events)

        # Update domain concepts
        self._update_domain_concepts(recent_events)

        # Evolve quality patterns
        self._evolve_quality_patterns(recent_events)

        # Update confidence scores
        self._update_confidence_scores()

        # Record evolution event
        evolution_event = {
            "timestamp": datetime.now().isoformat(),
            "events_processed": len(recent_events),
            "patterns_updated": len(self.patterns),
            "understanding_keys": len(self.understanding.domain_concepts),
            "quality_improvements": self._calculate_quality_improvements()
        }

        self.understanding.evolution_timeline.append(evolution_event)

        # Save updated state
        self._save_state()

        return {
            "refinement_complete": True,
            "events_processed": len(recent_events),
            "evolution_event": evolution_event,
            "improvements": self._calculate_quality_improvements()
        }

    def _analyze_contribution(self, contribution_data: Dict[str, Any]) -> List[str]:
        """Analyze a contribution to extract learned patterns."""
        patterns_learned = []

        # Extract code patterns
        if "files_changed" in contribution_data:
            for file_path in contribution_data["files_changed"]:
                file_patterns = self._extract_patterns_from_file(file_path)
                patterns_learned.extend(file_patterns)

        # Extract architectural patterns
        if "architecture_changes" in contribution_data:
            arch_patterns = self._extract_architectural_patterns(contribution_data["architecture_changes"])
            patterns_learned.extend(arch_patterns)

        return patterns_learned

    def _analyze_validation_results(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze validation results for insights."""
        insights = {
            "issues_found": len(validation_data.get("issues", [])),
            "warnings_found": len(validation_data.get("warnings", [])),
            "patterns_identified": [],
            "recommendations": []
        }

        # Analyze issues for patterns
        for issue in validation_data.get("issues", []):
            if "abstraction" in issue.lower():
                insights["patterns_identified"].append("over_abstraction")
            elif "dependency" in issue.lower():
                insights["patterns_identified"].append("unnecessary_dependency")
            elif "architecture" in issue.lower():
                insights["patterns_identified"].append("architecture_violation")

        return insights

    def _analyze_information_access(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how information is being accessed."""
        return {
            "context_type": usage_data.get("context_type"),
            "information_types": usage_data.get("information_types", []),
            "access_patterns": usage_data.get("access_patterns", []),
            "user_goals": self._infer_user_goals(usage_data)
        }

    def _analyze_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze the content of a file for context."""
        if not file_path.exists():
            return {"exists": False}

        try:
            content = file_path.read_text()
        except:
            return {"exists": True, "readable": False}

        analysis = {
            "exists": True,
            "readable": True,
            "file_type": file_path.suffix,
            "size": len(content),
            "lines": len(content.split('\n')),
            "language": self._detect_language(file_path),
            "imports": self._extract_imports(content),
            "classes": self._extract_classes(content),
            "functions": self._extract_functions(content),
            "patterns_used": self._identify_patterns_used(content),
            "architectural_role": self._determine_architectural_role(file_path, content)
        }

        return analysis

    def _find_relevant_patterns(self, file_analysis: Dict[str, Any], agent_id: str = None) -> List[CodePattern]:
        """Find patterns relevant to the file analysis, considering agent preferences."""
        relevant_patterns = []

        # Find patterns by language and type from core patterns
        language = file_analysis.get("language")
        classes = file_analysis.get("classes", [])
        functions = file_analysis.get("functions", [])

        # Get agent-specific preferences if available
        agent_preferences = {}
        if agent_id and agent_id in self.agent_understandings:
            agent_preferences = self.agent_understandings[agent_id].preferred_patterns

        for pattern in self.core_patterns.values():
            relevance_score = 0

            # Language match
            if pattern.context.get("language") == language:
                relevance_score += 0.3

            # Pattern type match
            if pattern.pattern_type in ["class", "function"]:
                if pattern.pattern_type == "class" and any(cls in pattern.signature for cls in classes):
                    relevance_score += 0.4
                elif pattern.pattern_type == "function" and any(func in pattern.signature for func in functions):
                    relevance_score += 0.4

            # Usage count bonus
            relevance_score += min(pattern.usage_count / 100, 0.2)

            # Quality bonus
            relevance_score += pattern.quality_score * 0.1

            # Agent preference bonus
            if pattern.pattern_id in agent_preferences:
                relevance_score += agent_preferences[pattern.pattern_id] * 0.3

            if relevance_score > 0.4:  # Relevance threshold
                # Mark as used to improve pattern quality
                pattern.usage_count += 1
                pattern.last_seen = datetime.now()
                relevant_patterns.append(pattern)

        # Sort by relevance (including agent preferences) and limit results
        relevant_patterns.sort(key=lambda p: (
            p.quality_score +
            min(p.usage_count / 100, 0.2) +
            agent_preferences.get(p.pattern_id, 0) * 0.3
        ), reverse=True)

        return relevant_patterns[:10]  # Top 10 most relevant

    def _get_understanding_context(self, file_analysis: Dict[str, Any], agent_id: str = None) -> Dict[str, Any]:
        """Get understanding context for the file."""
        context = {}

        # Get core domain concepts related to the file
        architectural_role = file_analysis.get("architectural_role")
        if architectural_role in self.core_understanding.domain_concepts:
            context["domain_knowledge"] = self.core_understanding.domain_concepts[architectural_role]

        # Get core architectural patterns
        patterns_used = file_analysis.get("patterns_used", [])
        for pattern in patterns_used:
            if pattern in self.core_understanding.architectural_patterns:
                context[f"architectural_{pattern}"] = self.core_understanding.architectural_patterns[pattern]

        # Get agent-specific understanding if available
        if agent_id and agent_id in self.agent_understandings:
            agent_understanding = self.agent_understandings[agent_id]

            # Add agent-specific successful approaches
            if agent_understanding.successful_approaches:
                context["agent_successful_approaches"] = agent_understanding.successful_approaches[:3]

            # Add agent-specific preferred patterns
            if agent_understanding.preferred_patterns:
                context["agent_preferred_patterns"] = list(agent_understanding.preferred_patterns.keys())[:5]

        return context

    def _get_quality_insights(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get quality insights for the file."""
        insights = {}

        # Check against learned quality patterns
        language = file_analysis.get("language")
        if language in self.understanding.quality_patterns:
            lang_patterns = self.understanding.quality_patterns[language]

            # Check for common issues
            insights["potential_issues"] = []
            insights["best_practices"] = lang_patterns.get("best_practices", [])
            insights["common_mistakes"] = lang_patterns.get("common_mistakes", [])

        # Get confidence in current patterns
        architectural_role = file_analysis.get("architectural_role")
        if architectural_role and architectural_role in self.understanding.confidence_scores:
            insights["confidence_score"] = self.understanding.confidence_scores[architectural_role]

        return insights

    def _get_evolution_context(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get evolution context showing how the codebase has changed."""
        context = {
            "recent_changes": [],
            "evolution_trends": [],
            "future_directions": []
        }

        # Get recent evolution events related to this file type
        file_type = file_analysis.get("architectural_role")
        recent_events = [e for e in self.understanding.evolution_timeline[-10:]
                        if file_type in str(e.get("context", ""))]

        context["recent_changes"] = recent_events

        # Identify trends
        if len(recent_events) >= 3:
            context["evolution_trends"] = self._analyze_evolution_trends(recent_events)

        return context

    def _update_patterns_from_event(self, event: LearningEvent):
        """Update core patterns based on a learning event."""
        for pattern_id in event.patterns_learned:
            if pattern_id in self.core_patterns:
                pattern = self.core_patterns[pattern_id]
                pattern.usage_count += 1
                pattern.last_seen = event.timestamp

                # Update quality based on event outcome
                quality_change = event.quality_metrics.get("contribution_quality", 0)
                pattern.quality_score = (pattern.quality_score + quality_change) / 2

                # Also update in core understanding if it's a universal pattern
                if pattern_id in self.core_understanding.universal_patterns:
                    self.core_understanding.universal_patterns[pattern_id] = pattern

    def _refine_understanding(self, event: LearningEvent):
        """Refine the system's understanding based on an event."""
        # Update core domain concepts (universal)
        if "domain_concepts" in event.context:
            for concept, data in event.context["domain_concepts"].items():
                if concept not in self.core_understanding.domain_concepts:
                    self.core_understanding.domain_concepts[concept] = data
                else:
                    # Merge/update existing concept
                    existing = self.core_understanding.domain_concepts[concept]
                    for key, value in data.items():
                        if key not in existing:
                            existing[key] = value

        # Update core architectural patterns (universal)
        if "architectural_patterns" in event.context:
            self.core_understanding.architectural_patterns.update(event.context["architectural_patterns"])

        # Update agent-specific understanding if agent is known
        agent_id = event.agent_id
        if agent_id and agent_id != "system":
            self._update_agent_specific_understanding(event)

    def _update_agent_specific_understanding(self, event: LearningEvent):
        """Update agent-specific understanding based on interaction."""
        agent_id = event.agent_id

        # Get or create agent-specific understanding
        if agent_id not in self.agent_understandings:
            self.agent_understandings[agent_id] = AgentSpecificUnderstanding(agent_id=agent_id)

        agent_understanding = self.agent_understandings[agent_id]

        # Update interaction history
        interaction_record = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "context": event.context,
            "outcome": event.outcome,
            "quality_metrics": event.quality_metrics
        }
        agent_understanding.learning_history.append(interaction_record)

        # Keep only recent history
        if len(agent_understanding.learning_history) > 20:
            agent_understanding.learning_history = agent_understanding.learning_history[-20:]

        # Update preferred patterns based on successful interactions
        if event.quality_metrics.get("contribution_quality", 0) > 0.8:
            for pattern_id in event.patterns_learned:
                if pattern_id not in agent_understanding.preferred_patterns:
                    agent_understanding.preferred_patterns[pattern_id] = 0.5
                # Increase preference for successful patterns
                agent_understanding.preferred_patterns[pattern_id] = min(
                    agent_understanding.preferred_patterns[pattern_id] + 0.1, 1.0
                )

        # Update successful approaches
        if "successful_approach" in event.outcome:
            approach = event.outcome["successful_approach"]
            if approach not in agent_understanding.successful_approaches:
                agent_understanding.successful_approaches.append(approach)

    def _learn_from_validation(self, event: LearningEvent):
        """Learn from validation event."""
        insights = event.outcome.get("insights", {})

        # Update quality patterns based on validation results
        for pattern in insights.get("patterns_identified", []):
            if pattern not in self.understanding.quality_patterns:
                self.understanding.quality_patterns[pattern] = {"occurrences": 0}
            self.understanding.quality_patterns[pattern]["occurrences"] += 1

    def _learn_from_usage(self, event: LearningEvent):
        """Learn from usage patterns."""
        access_patterns = event.outcome.get("access_patterns", {})

        # Update user workflow understanding
        context_type = access_patterns.get("context_type")
        if context_type:
            if context_type not in self.understanding.user_workflows:
                self.understanding.user_workflows[context_type] = []

            user_goals = access_patterns.get("user_goals", [])
            for goal in user_goals:
                if goal not in self.understanding.user_workflows[context_type]:
                    self.understanding.user_workflows[context_type].append(goal)

    def _load_state(self):
        """Load learning state from disk."""
        # Load core patterns
        if self.core_patterns_file.exists():
            try:
                with open(self.core_patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data.values():
                        # Convert timestamp strings back to datetime
                        if 'last_seen' in pattern_data and pattern_data['last_seen']:
                            pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                        pattern = CodePattern(**pattern_data)
                        self.core_patterns[pattern.pattern_id] = pattern
            except Exception as e:
                print(f"Error loading core patterns: {e}")

        # Load core understanding
        if self.core_understanding_file.exists():
            try:
                with open(self.core_understanding_file, 'r') as f:
                    data = json.load(f)
                    self.core_understanding = CoreUnderstanding(**data)
            except Exception as e:
                print(f"Error loading core understanding: {e}")

        # Load agent understandings
        if self.agent_understandings_file.exists():
            try:
                with open(self.agent_understandings_file, 'r') as f:
                    data = json.load(f)
                    for agent_id, understanding_data in data.items():
                        understanding = AgentSpecificUnderstanding(**understanding_data)
                        self.agent_understandings[agent_id] = understanding
            except Exception as e:
                print(f"Error loading agent understandings: {e}")

        # Load learning events (last 1000)
        if self.learning_events_file.exists():
            try:
                with open(self.learning_events_file, 'r') as f:
                    data = json.load(f)
                    for event_data in data[-1000:]:  # Keep last 1000 events
                        # Convert timestamp
                        if 'timestamp' in event_data:
                            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        event = LearningEvent(**event_data)
                        self.learning_events.append(event)
            except Exception as e:
                print(f"Error loading learning events: {e}")

    def _save_state(self):
        """Save learning state to disk."""
        # Save core patterns
        core_patterns_data = {}
        for pattern_id, pattern in self.core_patterns.items():
            pattern_data = {
                **pattern.__dict__,
                'last_seen': pattern.last_seen.isoformat() if pattern.last_seen else None,
                'related_patterns': list(pattern.related_patterns)
            }
            core_patterns_data[pattern_id] = pattern_data

        with open(self.core_patterns_file, 'w') as f:
            json.dump(core_patterns_data, f, indent=2)

        # Save core understanding
        with open(self.core_understanding_file, 'w') as f:
            json.dump(self.core_understanding.__dict__, f, indent=2, default=str)

        # Save agent understandings
        agent_data = {}
        for agent_id, understanding in self.agent_understandings.items():
            agent_data[agent_id] = understanding.__dict__

        with open(self.agent_understandings_file, 'w') as f:
            json.dump(agent_data, f, indent=2)

        # Save learning events (keep last 1000)
        events_data = []
        for event in self.learning_events[-1000:]:
            event_data = {
                **event.__dict__,
                'timestamp': event.timestamp.isoformat(),
                'related_patterns': list(event.patterns_learned) if hasattr(event, 'patterns_learned') else []
            }
            events_data.append(event_data)

        with open(self.learning_events_file, 'w') as f:
            json.dump(events_data, f, indent=2)

    def _save_learning_events(self):
        """Save learning events to disk."""
        events_data = []
        for event in self.learning_events[-1000:]:
            event_data = event.__dict__.copy()
            event_data['timestamp'] = event.timestamp.isoformat()
            events_data.append(event_data)

        with open(self.learning_events_file, 'w') as f:
            json.dump(events_data, f, indent=2)

    # Helper methods for analysis
    def _extract_patterns_from_file(self, file_path: str) -> List[str]:
        """Extract patterns from a file."""
        patterns = []
        try:
            content = Path(file_path).read_text()

            # Extract class patterns
            class_matches = re.findall(r'class (\w+).*?:', content)
            for class_name in class_matches:
                pattern_id = f"class_{class_name}"
                patterns.append(pattern_id)

                # Store the pattern
                self.patterns[pattern_id] = CodePattern(
                    pattern_id=pattern_id,
                    pattern_type="class",
                    signature=f"class {class_name}",
                    context={"language": "python", "file": file_path},
                    last_seen=datetime.now()
                )

            # Extract function patterns
            func_matches = re.findall(r'def (\w+)\s*\(', content)
            for func_name in func_matches:
                pattern_id = f"func_{func_name}"
                patterns.append(pattern_id)

                self.patterns[pattern_id] = CodePattern(
                    pattern_id=pattern_id,
                    pattern_type="function",
                    signature=f"def {func_name}(",
                    context={"language": "python", "file": file_path},
                    last_seen=datetime.now()
                )

        except:
            pass

        return patterns

    def _detect_language(self, file_path: Path) -> str:
        """Detect the programming language of a file."""
        if file_path.suffix == '.py':
            return 'python'
        elif file_path.suffix == '.js':
            return 'javascript'
        elif file_path.suffix == '.ts':
            return 'typescript'
        elif file_path.suffix in ['.md', '.txt']:
            return 'markdown'
        elif file_path.suffix == '.json':
            return 'json'
        else:
            return 'unknown'

    def _extract_imports(self, content: str) -> List[str]:
        """Extract imports from code."""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        return imports

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from code."""
        return re.findall(r'class (\w+)', content)

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from code."""
        return re.findall(r'def (\w+)\s*\(', content)

    def _identify_patterns_used(self, content: str) -> List[str]:
        """Identify architectural patterns used in the code."""
        patterns = []

        # Simple pattern detection
        if 'CommandBus' in content:
            patterns.append('command_pattern')
        if 'ApplicationFacade' in content:
            patterns.append('facade_pattern')
        if 'repository' in content.lower():
            patterns.append('repository_pattern')
        if 'BlockProcessor' in content:
            patterns.append('strategy_pattern')

        return patterns

    def _determine_architectural_role(self, file_path: Path, content: str) -> str:
        """Determine the architectural role of a file."""
        path_str = str(file_path)

        if 'domain' in path_str:
            return 'domain_layer'
        elif 'application' in path_str:
            return 'application_layer'
        elif 'infrastructure' in path_str:
            return 'infrastructure_layer'
        elif 'ui' in path_str or 'qt' in path_str:
            return 'presentation_layer'
        elif 'test' in path_str:
            return 'test_layer'
        else:
            return 'utility'

    def _calculate_context_confidence(self, patterns: List[CodePattern], understanding: Dict) -> float:
        """Calculate confidence score for provided context."""
        if not patterns:
            return 0.0

        # Base confidence from pattern quality
        pattern_quality = sum(p.quality_score for p in patterns) / len(patterns)

        # Bonus for understanding depth
        understanding_depth = min(len(understanding) / 10, 1.0)

        # Usage bonus
        usage_bonus = min(sum(p.usage_count for p in patterns) / 1000, 0.2)

        return min(pattern_quality + understanding_depth + usage_bonus, 1.0)

    # Additional helper methods would be implemented...
    def _assess_contribution_quality(self, data: Dict) -> Dict[str, float]:
        return {"contribution_quality": 0.8}  # Placeholder

    def _extract_architectural_patterns(self, changes: Dict) -> List[str]:
        return []  # Placeholder

    def _extract_validation_metrics(self, data: Dict) -> Dict[str, float]:
        return {}  # Placeholder

    def _infer_user_goals(self, data: Dict) -> List[str]:
        return []  # Placeholder

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        return {"query": query}  # Placeholder

    def _find_patterns_for_query(self, analysis: Dict) -> List[CodePattern]:
        return []  # Placeholder

    def _get_understanding_for_query(self, analysis: Dict) -> Dict[str, Any]:
        return {}  # Placeholder

    def _get_historical_context(self, analysis: Dict) -> Dict[str, Any]:
        return {}  # Placeholder

    def _calculate_query_confidence(self, analysis: Dict, patterns: List) -> float:
        return 0.5  # Placeholder

    def _get_cursor_specific_context(self, file_path: Path, cursor_pos: Dict) -> Dict[str, Any]:
        return {}  # Placeholder

    def _update_pattern_quality_scores(self, events: List[LearningEvent]):
        pass  # Placeholder

    def _refine_architectural_understanding(self, events: List[LearningEvent]):
        pass  # Placeholder

    def _update_domain_concepts(self, events: List[LearningEvent]):
        pass  # Placeholder

    def _evolve_quality_patterns(self, events: List[LearningEvent]):
        pass  # Placeholder

    def _update_confidence_scores(self):
        pass  # Placeholder

    def _calculate_quality_improvements(self) -> Dict[str, Any]:
        return {}  # Placeholder

    def _analyze_evolution_trends(self, events: List) -> List[str]:
        return []  # Placeholder

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        total_patterns = len(self.patterns)
        total_events = len(self.learning_events)

        # Calculate pattern quality distribution
        quality_scores = [p.quality_score for p in self.patterns.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Recent activity (last 7 days)
        recent_events = [e for e in self.learning_events
                        if e.timestamp > datetime.now() - timedelta(days=7)]

        # Pattern usage stats
        usage_stats = {}
        for pattern in self.patterns.values():
            usage_stats[pattern.pattern_type] = usage_stats.get(pattern.pattern_type, 0) + pattern.usage_count

        return {
            "total_patterns": total_patterns,
            "total_events": total_events,
            "average_pattern_quality": avg_quality,
            "recent_events": len(recent_events),
            "pattern_usage_by_type": usage_stats,
            "understanding_domains": len(self.understanding.domain_concepts),
            "last_refinement": self.understanding.evolution_timeline[-1] if self.understanding.evolution_timeline else None
        }


def main():
    """CLI interface for the learning engine."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python learning_engine.py refine          # Refine understanding")
        print("  python learning_engine.py stats           # Show learning statistics")
        print("  python learning_engine.py context <file>  # Get context for file")
        print("  python learning_engine.py record <type> <data>  # Record event")
        sys.exit(1)

    engine = SelfRefiningLearningEngine()
    command = sys.argv[1]

    if command == "refine":
        results = engine.refine_understanding()
        print("Refinement Results:")
        print(json.dumps(results, indent=2))

    elif command == "stats":
        stats = engine.get_learning_stats()
        print("Learning Statistics:")
        print(json.dumps(stats, indent=2, default=str))

    elif command == "context" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        cursor_pos = None
        if len(sys.argv) > 3:
            try:
                cursor_pos = {"line": int(sys.argv[3])}
            except ValueError:
                pass

        context = engine.get_context_for_file(file_path, cursor_pos)
        print("File Context:")
        print(json.dumps(context, indent=2))

    elif command == "record" and len(sys.argv) > 3:
        event_type = sys.argv[2]
        event_data = json.loads(sys.argv[3])

        if event_type == "contribution":
            result = engine.record_contribution_event(
                event_data.get("agent_id", "unknown"),
                event_data
            )
            print(f"Contribution recorded: {result}")
        elif event_type == "validation":
            result = engine.record_validation_event(
                event_data.get("agent_id", "unknown"),
                event_data
            )
            print(f"Validation recorded: {result}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
