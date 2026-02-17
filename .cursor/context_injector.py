#!/usr/bin/env python3
"""
Cursor IDE Context Injector for AgentAssets.

Automatically provides relevant AgentAssets context when AI agents work with files/tabs
in Cursor IDE, using the learning engine to deliver personalized, relevant information.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading


@dataclass
class TabContext:
    """Represents context for a specific tab."""
    tab_id: str
    file_path: str
    cursor_position: Optional[Dict[str, int]] = None
    content_hash: str = ""
    last_context_update: datetime = None
    context_cache: Dict[str, Any] = None
    relevance_score: float = 0.0


@dataclass
class ContextRequest:
    """Represents a request for context."""
    tab_id: str
    file_path: str
    cursor_position: Optional[Dict[str, int]] = None
    request_type: str = "auto"  # auto, manual, on_demand
    context_types: List[str] = None
    agent_id: str = "unknown"

    def __post_init__(self):
        if self.context_types is None:
            self.context_types = ["patterns", "understanding", "quality", "evolution"]


class CursorContextInjector:
    """Injects relevant AgentAssets context into Cursor IDE tabs."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.agent_assets_root = self.project_root / "AgentAssets"
        self.data_dir = self.agent_assets_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Import learning engine
        sys.path.insert(0, str(self.agent_assets_root / "scripts"))
        try:
            from learning_engine import SelfRefiningLearningEngine
            self.learning_engine = SelfRefiningLearningEngine()
        except ImportError:
            print("Warning: Learning engine not available")
            self.learning_engine = None

        # Context tracking
        self.active_tabs: Dict[str, TabContext] = {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 300  # 5 minutes

        # Background thread for context updates
        self.update_thread = None
        self.running = False

    def start_context_injection(self):
        """Start the context injection system."""
        print("🧠 Starting Cursor Context Injector...")

        if self.learning_engine is None:
            print("❌ Learning engine not available - context injection disabled")
            return False

        self.running = True
        self.update_thread = threading.Thread(target=self._background_context_updater)
        self.update_thread.daemon = True
        self.update_thread.start()

        print("✅ Context injector started")
        return True

    def stop_context_injection(self):
        """Stop the context injection system."""
        print("🛑 Stopping context injector...")
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("✅ Context injector stopped")

    def get_context_for_tab(self, request: ContextRequest) -> Dict[str, Any]:
        """
        Get context for a specific tab.

        This is the main entry point for Cursor IDE to request context.
        """
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_context = self._get_cached_context(cache_key, request)

        if cached_context:
            # Record usage for learning
            if self.learning_engine:
                self.learning_engine.record_usage_event(request.agent_id, {
                    "context_type": "tab_context",
                    "file_path": request.file_path,
                    "cached": True,
                    "relevance_score": cached_context.get("relevance_score", 0)
                })
            return cached_context

        # Generate new context
        context = self._generate_tab_context(request)

        # Cache the context
        self._cache_context(cache_key, context)

        # Update tab tracking
        self._update_tab_context(request.tab_id, request.file_path, context)

        # Record usage for learning
        if self.learning_engine:
            self.learning_engine.record_usage_event(request.agent_id, {
                "context_type": "tab_context",
                "file_path": request.file_path,
                "cached": False,
                "relevance_score": context.get("relevance_score", 0),
                "context_types": request.context_types
            })

        return context

    def get_context_for_selection(self, tab_id: str, selection: Dict[str, Any], agent_id: str = "unknown") -> Dict[str, Any]:
        """
        Get context for a specific text selection in a tab.
        """
        if tab_id not in self.active_tabs:
            return {"error": "Tab not found"}

        tab_context = self.active_tabs[tab_id]

        # Analyze the selection
        selection_context = self._analyze_selection(tab_context.file_path, selection)

        # Get relevant context
        context = {
            "selection_analysis": selection_context,
            "file_context": tab_context.context_cache,
            "relevant_patterns": self._find_selection_patterns(selection_context),
            "suggested_actions": self._suggest_selection_actions(selection_context),
            "confidence_score": self._calculate_selection_confidence(selection_context)
        }

        # Record usage
        if self.learning_engine:
            self.learning_engine.record_usage_event(agent_id, {
                "context_type": "selection_context",
                "file_path": tab_context.file_path,
                "selection": selection,
                "analysis": selection_context
            })

        return context

    def get_predictive_context(self, tab_id: str, current_input: str, agent_id: str = "unknown") -> Dict[str, Any]:
        """
        Get predictive context based on current input and cursor position.
        """
        if tab_id not in self.active_tabs:
            return {"error": "Tab not found"}

        tab_context = self.active_tabs[tab_id]

        # Analyze current input for predictions
        predictions = self._analyze_input_predictions(tab_context.file_path, current_input)

        context = {
            "predictions": predictions,
            "file_context": tab_context.context_cache,
            "likely_intent": self._infer_intent(predictions),
            "suggested_completions": predictions.get("completions", []),
            "relevant_examples": self._find_relevant_examples(predictions)
        }

        return context

    def refresh_tab_context(self, tab_id: str) -> bool:
        """Force refresh context for a specific tab."""
        if tab_id not in self.active_tabs:
            return False

        tab_context = self.active_tabs[tab_id]

        # Clear cache for this tab
        cache_key = self._generate_cache_key_from_tab(tab_context)
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]

        # Generate new context
        request = ContextRequest(
            tab_id=tab_id,
            file_path=tab_context.file_path,
            cursor_position=tab_context.cursor_position,
            request_type="refresh"
        )

        new_context = self._generate_tab_context(request)
        self._cache_context(cache_key, new_context)
        self._update_tab_context(tab_id, tab_context.file_path, new_context)

        return True

    def _generate_tab_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Generate comprehensive context for a tab."""
        if not self.learning_engine:
            return {"error": "Learning engine not available"}

        # Get base context from learning engine
        base_context = self.learning_engine.get_context_for_file(
            request.file_path,
            request.cursor_position
        )

        # Enhance with tab-specific information
        tab_context = {
            "tab_id": request.tab_id,
            "file_path": request.file_path,
            "cursor_position": request.cursor_position,
            "request_type": request.request_type,
            "generated_at": datetime.now().isoformat(),
            "context_types_provided": request.context_types,
            **base_context
        }

        # Add tab-specific enhancements
        tab_context["navigation_suggestions"] = self._generate_navigation_suggestions(tab_context)
        tab_context["related_files"] = self._find_related_files(request.file_path)
        tab_context["recent_changes"] = self._get_recent_changes(request.file_path)
        tab_context["collaboration_hints"] = self._generate_collaboration_hints(tab_context)

        # Calculate overall relevance score
        tab_context["relevance_score"] = self._calculate_overall_relevance(tab_context)

        return tab_context

    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate a cache key for a context request."""
        key_components = [
            request.file_path,
            str(request.cursor_position or {}),
            str(sorted(request.context_types)),
            request.request_type
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_cache_key_from_tab(self, tab_context: TabContext) -> str:
        """Generate cache key from tab context."""
        key_components = [
            tab_context.file_path,
            str(tab_context.cursor_position or {}),
            tab_context.content_hash
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_context(self, cache_key: str, request: ContextRequest) -> Optional[Dict[str, Any]]:
        """Get context from cache if valid."""
        if cache_key not in self.context_cache:
            return None

        cached = self.context_cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get("generated_at", "2000-01-01T00:00:00"))

        # Check if cache is still valid
        if datetime.now() - cached_time > timedelta(seconds=self.cache_timeout):
            del self.context_cache[cache_key]
            return None

        # Check if file has changed
        if not self._file_changed_since_cache(request.file_path, cached_time):
            return cached

        # File changed, invalidate cache
        del self.context_cache[cache_key]
        return None

    def _cache_context(self, cache_key: str, context: Dict[str, Any]):
        """Cache context data."""
        self.context_cache[cache_key] = context

        # Limit cache size
        if len(self.context_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self.context_cache.items(),
                                key=lambda x: x[1].get("generated_at", "2000-01-01"))
            to_remove = sorted_cache[:20]  # Remove 20 oldest
            for key, _ in to_remove:
                del self.context_cache[key]

    def _update_tab_context(self, tab_id: str, file_path: str, context: Dict[str, Any]):
        """Update tracking for a tab."""
        content_hash = self._calculate_file_hash(file_path)

        tab_context = TabContext(
            tab_id=tab_id,
            file_path=file_path,
            content_hash=content_hash,
            last_context_update=datetime.now(),
            context_cache=context,
            relevance_score=context.get("relevance_score", 0.0)
        )

        self.active_tabs[tab_id] = tab_context

    def _file_changed_since_cache(self, file_path: str, cache_time: datetime) -> bool:
        """Check if file has changed since cache time."""
        try:
            file_stat = Path(file_path).stat()
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
            return file_mtime > cache_time
        except:
            return True  # Assume changed if we can't check

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def _background_context_updater(self):
        """Background thread for context updates."""
        while self.running:
            try:
                # Update contexts for active tabs
                for tab_id, tab_context in list(self.active_tabs.items()):
                    if self._should_refresh_tab_context(tab_context):
                        self.refresh_tab_context(tab_id)

                # Periodic learning engine refinement
                if int(time.time()) % 3600 == 0:  # Every hour
                    if self.learning_engine:
                        self.learning_engine.refine_understanding()

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Context updater error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _should_refresh_tab_context(self, tab_context: TabContext) -> bool:
        """Determine if tab context should be refreshed."""
        if not tab_context.last_context_update:
            return True

        time_since_update = datetime.now() - tab_context.last_context_update
        return time_since_update.total_seconds() > 600  # Refresh every 10 minutes

    # Context enhancement methods
    def _generate_navigation_suggestions(self, tab_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate navigation suggestions based on context."""
        suggestions = []

        file_analysis = tab_context.get("file_analysis", {})

        # Suggest related files
        if file_analysis.get("architectural_role") == "domain_layer":
            suggestions.append({
                "type": "related_file",
                "description": "View application service",
                "path": file_analysis.get("file_path", "").replace("domain", "application")
            })

        # Suggest documentation
        if file_analysis.get("language") == "python":
            suggestions.append({
                "type": "documentation",
                "description": "View API documentation",
                "path": f"docs/encyclopedia/03-application/{file_analysis.get('file_path', '').split('/')[-1].replace('.py', '.md')}"
            })

        return suggestions

    def _find_related_files(self, file_path: str) -> List[Dict[str, Any]]:
        """Find files related to the current file."""
        related = []
        path = Path(file_path)

        # Find test files
        if "src" in str(path):
            test_path = str(path).replace("src", "tests")
            test_file = Path(test_path.replace(".py", "_test.py"))
            if test_file.exists():
                related.append({
                    "type": "test",
                    "path": str(test_file),
                    "description": "Test file"
                })

        # Find interface files
        if path.suffix == ".py":
            interface_file = path.parent / f"I{path.stem}.py"
            if interface_file.exists():
                related.append({
                    "type": "interface",
                    "path": str(interface_file),
                    "description": "Interface definition"
                })

        return related

    def _get_recent_changes(self, file_path: str) -> List[Dict[str, Any]]:
        """Get recent changes to the file."""
        # This would integrate with git
        # For now, return mock data
        return [
            {
                "type": "modification",
                "description": "Updated method signatures",
                "timestamp": datetime.now().isoformat(),
                "author": "AI Agent"
            }
        ]

    def _generate_collaboration_hints(self, tab_context: Dict[str, Any]) -> List[str]:
        """Generate collaboration hints based on context."""
        hints = []

        quality_insights = tab_context.get("quality_insights", {})

        if quality_insights.get("potential_issues"):
            hints.append("Consider addressing quality issues before committing")

        if tab_context.get("confidence_score", 0) > 0.8:
            hints.append("High confidence context available - good patterns detected")

        return hints

    def _calculate_overall_relevance(self, tab_context: Dict[str, Any]) -> float:
        """Calculate overall relevance score for context."""
        scores = []

        # Base relevance from learning engine
        base_score = tab_context.get("confidence_score", 0)
        if base_score > 0:
            scores.append(base_score)

        # Relevance from patterns
        patterns = tab_context.get("relevant_patterns", [])
        if patterns:
            pattern_score = min(len(patterns) / 10, 1.0)  # Up to 10 patterns
            scores.append(pattern_score)

        # Relevance from understanding
        understanding = tab_context.get("understanding_context", {})
        if understanding:
            understanding_score = min(len(understanding) / 5, 1.0)  # Up to 5 understanding items
            scores.append(understanding_score)

        return sum(scores) / len(scores) if scores else 0.0

    # Selection and prediction methods
    def _analyze_selection(self, file_path: str, selection: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a text selection."""
        # This would analyze the selected text
        return {
            "selected_text": selection.get("text", ""),
            "start_line": selection.get("start", {}).get("line", 0),
            "end_line": selection.get("end", {}).get("line", 0),
            "language_constructs": [],  # Would analyze for classes, functions, etc.
            "intent": "unknown"
        }

    def _find_selection_patterns(self, selection_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns relevant to the selection."""
        return []  # Implementation would search patterns

    def _suggest_selection_actions(self, selection_analysis: Dict[str, Any]) -> List[str]:
        """Suggest actions for the selection."""
        return ["Extract method", "Add documentation", "Create test"]

    def _calculate_selection_confidence(self, selection_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in selection analysis."""
        return 0.7

    def _analyze_input_predictions(self, file_path: str, current_input: str) -> Dict[str, Any]:
        """Analyze current input for predictions."""
        return {
            "likely_next": [],
            "completions": [],
            "context_aware_suggestions": []
        }

    def _infer_intent(self, predictions: Dict[str, Any]) -> str:
        """Infer user intent from predictions."""
        return "unknown"


# Global instance
_context_injector = None


def get_context_injector() -> CursorContextInjector:
    """Get the global context injector instance."""
    global _context_injector
    if _context_injector is None:
        _context_injector = CursorContextInjector()
    return _context_injector


def start_context_injection():
    """Start the context injection system."""
    injector = get_context_injector()
    return injector.start_context_injection()


def stop_context_injection():
    """Stop the context injection system."""
    injector = get_context_injector()
    injector.stop_context_injection()


def get_tab_context(tab_id: str, file_path: str, cursor_position: Dict[str, int] = None,
                   agent_id: str = "unknown") -> Dict[str, Any]:
    """Get context for a tab."""
    injector = get_context_injector()
    request = ContextRequest(
        tab_id=tab_id,
        file_path=file_path,
        cursor_position=cursor_position,
        agent_id=agent_id
    )
    return injector.get_context_for_tab(request)


def get_selection_context(tab_id: str, selection: Dict[str, Any], agent_id: str = "unknown") -> Dict[str, Any]:
    """Get context for a selection."""
    injector = get_context_injector()
    return injector.get_context_for_selection(tab_id, selection, agent_id)


def get_predictive_context(tab_id: str, current_input: str, agent_id: str = "unknown") -> Dict[str, Any]:
    """Get predictive context."""
    injector = get_context_injector()
    return injector.get_predictive_context(tab_id, current_input, agent_id)


# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        start_context_injection()
        print("Context injector started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_context_injection()
    else:
        print("Usage: python context_injector.py start")

