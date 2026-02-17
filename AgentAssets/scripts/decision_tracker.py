#!/usr/bin/env python3
"""
AI Agent Council Decision Tracker

Tracks and analyzes council decisions to improve decision-making effectiveness.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DecisionTracker:
    """Tracks council decisions and analyzes patterns."""

    def __init__(self):
        self.data_dir = Path("AgentAssets/data")
        self.decisions_file = self.data_dir / "council_decisions.json"
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        self.data_dir.mkdir(exist_ok=True)
        if not self.decisions_file.exists():
            with open(self.decisions_file, 'w') as f:
                json.dump([], f)

    def record_decision(self, decision_type: str, title: str, description: str,
                       council_votes: Dict[str, str], unanimous_decision: str,
                       implemented: bool = False) -> None:
        """Record a council decision."""
        decisions = self.load_decisions()

        decision = {
            "id": f"{decision_type}_{len(decisions)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "type": decision_type,
            "title": title,
            "description": description,
            "council_votes": council_votes,
            "unanimous_decision": unanimous_decision,
            "implemented": implemented
        }

        decisions.append(decision)
        self.save_decisions(decisions)

    def load_decisions(self) -> List[Dict]:
        """Load all decisions."""
        try:
            with open(self.decisions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_decisions(self, decisions: List[Dict]) -> None:
        """Save decisions to file."""
        with open(self.decisions_file, 'w') as f:
            json.dump(decisions, f, indent=2)

    def analyze_decisions(self) -> Dict:
        """Analyze decision patterns."""
        decisions = self.load_decisions()

        analysis = {
            "total_decisions": len(decisions),
            "implemented_count": sum(1 for d in decisions if d.get("implemented", False)),
            "by_type": {},
            "success_rate": 0.0
        }

        if decisions:
            analysis["success_rate"] = analysis["implemented_count"] / analysis["total_decisions"]

        for decision in decisions:
            decision_type = decision.get("type", "unknown")
            analysis["by_type"][decision_type] = analysis["by_type"].get(decision_type, 0) + 1

        return analysis


def main():
    """CLI interface."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python decision_tracker.py [record|analyze|list]")
        return

    tracker = DecisionTracker()
    command = sys.argv[1]

    if command == "record":
        if len(sys.argv) < 6:
            print("Usage: python decision_tracker.py record <type> <title> <description> <unanimous_decision>")
            return

        decision_type = sys.argv[2]
        title = sys.argv[3]
        description = sys.argv[4]
        unanimous_decision = sys.argv[5]

        # For simplicity, create mock council votes
        council_votes = {
            "Architect": "approve",
            "Systems": "approve",
            "UX": "approve",
            "Pragmatic": "approve"
        }

        tracker.record_decision(decision_type, title, description, council_votes, unanimous_decision)

    elif command == "analyze":
        analysis = tracker.analyze_decisions()
        print("Decision Analysis:")
        print(f"Total Decisions: {analysis['total_decisions']}")
        print(f"Implemented: {analysis['implemented_count']}")
        print(f"Success Rate: {analysis['success_rate']:.1%}")
        print("By Type:")
        for decision_type, count in analysis["by_type"].items():
            print(f"  {decision_type}: {count}")

    elif command == "list":
        decisions = tracker.load_decisions()
        for decision in decisions[-5:]:  # Show last 5
            print(f"{decision['timestamp'][:10]} {decision['type']}: {decision['title']}")
            print(f"  Decision: {decision['unanimous_decision']}")
            print(f"  Implemented: {decision.get('implemented', False)}")
            print()


if __name__ == "__main__":
    main()