#!/usr/bin/env python3
"""
AI Agent Feedback System for continuous improvement.

This system tracks AI agent performance, identifies improvement areas,
and provides personalized feedback to enhance collaboration quality.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter


@dataclass
class ContributionRecord:
    """Record of an AI agent's contribution."""
    id: str
    agent_id: str
    timestamp: datetime
    contribution_type: str  # feature, refactor, documentation, review, etc.
    files_changed: List[str]
    quality_score: float  # 0.0 to 1.0
    review_feedback: Optional[str] = None
    core_values_alignment: float = 0.5  # How well it aligns with core values
    architecture_compliance: float = 0.5  # How well it follows architecture
    implementation_quality: float = 0.5  # Code quality score
    documentation_quality: float = 0.5  # Documentation completeness
    tags: List[str] = None  # Additional categorization tags


@dataclass
class AgentProfile:
    """Profile tracking an AI agent's performance and improvement areas."""
    agent_id: str
    total_contributions: int = 0
    average_quality_score: float = 0.0
    strength_areas: List[str] = None
    improvement_areas: List[str] = None
    contribution_types: Dict[str, int] = None
    common_feedback: List[str] = None
    learning_progress: Dict[str, float] = None  # Track improvement over time
    last_updated: datetime = None

    def __post_init__(self):
        if self.strength_areas is None:
            self.strength_areas = []
        if self.improvement_areas is None:
            self.improvement_areas = []
        if self.contribution_types is None:
            self.contribution_types = {}
        if self.common_feedback is None:
            self.common_feedback = []
        if self.learning_progress is None:
            self.learning_progress = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()


class AIFeedbackSystem:
    """System for tracking and improving AI agent performance."""

    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.contributions_file = self.data_dir / "contributions.json"
        self.profiles_file = self.data_dir / "agent_profiles.json"

    def record_contribution(self, record: ContributionRecord) -> str:
        """Record a new AI agent contribution."""
        record.id = str(uuid.uuid4())
        record.timestamp = datetime.now()

        contributions = self._load_contributions()
        contributions.append(record)
        self._save_contributions(contributions)

        # Update agent profile
        self._update_agent_profile(record)

        return record.id

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get performance profile for an AI agent."""
        profiles = self._load_profiles()
        return profiles.get(agent_id)

    def generate_feedback_report(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate a personalized feedback report for an agent."""
        contributions = self._get_recent_contributions(agent_id, days)

        if not contributions:
            return {"message": f"No contributions found for {agent_id} in the last {days} days"}

        profile = self.get_agent_profile(agent_id)

        report = {
            "agent_id": agent_id,
            "period_days": days,
            "total_contributions": len(contributions),
            "average_quality_score": sum(c.quality_score for c in contributions) / len(contributions),
            "contribution_breakdown": self._analyze_contribution_types(contributions),
            "quality_trends": self._analyze_quality_trends(contributions),
            "strengths": profile.strength_areas if profile else [],
            "improvement_areas": profile.improvement_areas if profile else [],
            "personalized_recommendations": self._generate_recommendations(contributions, profile),
            "common_feedback_themes": self._extract_feedback_themes(contributions),
            "learning_suggestions": self._suggest_learning_activities(contributions)
        }

        return report

    def get_top_performers(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top-performing AI agents by average quality score."""
        profiles = self._load_profiles()

        scored_agents = []
        for agent_id, profile in profiles.items():
            if profile.total_contributions >= 3:  # Minimum contributions for ranking
                scored_agents.append((agent_id, profile.average_quality_score))

        return sorted(scored_agents, key=lambda x: x[1], reverse=True)[:limit]

    def identify_training_needs(self) -> Dict[str, Any]:
        """Analyze overall system to identify common training needs."""
        all_contributions = self._load_contributions()
        all_profiles = self._load_profiles()

        if not all_contributions:
            return {"message": "No contribution data available"}

        analysis = {
            "system_wide_issues": self._analyze_system_issues(all_contributions),
            "common_improvement_areas": self._find_common_weaknesses(all_profiles),
            "training_recommendations": self._generate_training_recommendations(all_contributions),
            "quality_distribution": self._analyze_quality_distribution(all_contributions),
            "timestamp": datetime.now().isoformat()
        }

        return analysis

    def _load_contributions(self) -> List[ContributionRecord]:
        """Load contribution records from storage."""
        if not self.contributions_file.exists():
            return []

        try:
            with open(self.contributions_file, 'r') as f:
                data = json.load(f)
                contributions = []
                for item in data:
                    # Handle timestamp parsing
                    if isinstance(item['timestamp'], str):
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    contributions.append(ContributionRecord(**item))
                return contributions
        except (json.JSONDecodeError, KeyError):
            return []

    def _save_contributions(self, contributions: List[ContributionRecord]):
        """Save contribution records to storage."""
        data = []
        for contrib in contributions:
            item = asdict(contrib)
            # Convert datetime to string
            item['timestamp'] = item['timestamp'].isoformat()
            data.append(item)

        with open(self.contributions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_profiles(self) -> Dict[str, AgentProfile]:
        """Load agent profiles from storage."""
        if not self.profiles_file.exists():
            return {}

        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                profiles = {}
                for agent_id, profile_data in data.items():
                    # Handle timestamp parsing
                    if isinstance(profile_data.get('last_updated'), str):
                        profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                    profiles[agent_id] = AgentProfile(**profile_data)
                return profiles
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_profiles(self, profiles: Dict[str, AgentProfile]):
        """Save agent profiles to storage."""
        data = {}
        for agent_id, profile in profiles.items():
            item = asdict(profile)
            # Convert datetime to string
            item['last_updated'] = item['last_updated'].isoformat()
            data[agent_id] = item

        with open(self.profiles_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_agent_profile(self, contribution: ContributionRecord):
        """Update or create agent profile based on contribution."""
        profiles = self._load_profiles()

        if contribution.agent_id not in profiles:
            profiles[contribution.agent_id] = AgentProfile(agent_id=contribution.agent_id)

        profile = profiles[contribution.agent_id]

        # Update statistics
        total_score = profile.average_quality_score * profile.total_contributions
        profile.total_contributions += 1
        profile.average_quality_score = (total_score + contribution.quality_score) / profile.total_contributions

        # Update contribution types
        profile.contribution_types[contribution.contribution_type] = \
            profile.contribution_types.get(contribution.contribution_type, 0) + 1

        # Add feedback if available
        if contribution.review_feedback:
            profile.common_feedback.append(contribution.review_feedback)

        # Update strengths and improvement areas
        profile.strengths, profile.improvement_areas = self._analyze_agent_skills(contribution, profile)

        profile.last_updated = datetime.now()

        self._save_profiles(profiles)

    def _get_recent_contributions(self, agent_id: str, days: int) -> List[ContributionRecord]:
        """Get contributions from an agent within the specified days."""
        all_contributions = self._load_contributions()
        cutoff_date = datetime.now() - timedelta(days=days)

        return [c for c in all_contributions
                if c.agent_id == agent_id and c.timestamp >= cutoff_date]

    def _analyze_contribution_types(self, contributions: List[ContributionRecord]) -> Dict[str, Any]:
        """Analyze distribution of contribution types."""
        type_counts = Counter(c.contribution_type for c in contributions)
        avg_scores_by_type = {}

        for contrib_type in type_counts.keys():
            type_contributions = [c for c in contributions if c.contribution_type == contrib_type]
            avg_scores_by_type[contrib_type] = \
                sum(c.quality_score for c in type_contributions) / len(type_contributions)

        return {
            "counts": dict(type_counts),
            "average_scores": avg_scores_by_type,
            "most_common": type_counts.most_common(1)[0][0] if type_counts else None
        }

    def _analyze_quality_trends(self, contributions: List[ContributionRecord]) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if len(contributions) < 2:
            return {"message": "Need at least 2 contributions to analyze trends"}

        # Sort by timestamp
        sorted_contribs = sorted(contributions, key=lambda c: c.timestamp)

        # Calculate trend (simple linear regression slope)
        n = len(sorted_contribs)
        x = list(range(n))  # Time indices
        y = [c.quality_score for c in sorted_contribs]

        # Simple trend calculation
        if n > 1:
            slope = (y[-1] - y[0]) / (n - 1)
            trend = "improving" if slope > 0.05 else "declining" if slope < -0.05 else "stable"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "slope": slope if 'slope' in locals() else 0,
            "first_score": y[0],
            "latest_score": y[-1],
            "consistency": len([s for s in y if s >= 0.8]) / len(y)  # % of high-quality contributions
        }

    def _analyze_agent_skills(self, contribution: ContributionRecord,
                            profile: AgentProfile) -> Tuple[List[str], List[str]]:
        """Analyze an agent's strengths and improvement areas."""
        strengths = []
        improvements = []

        # Analyze based on contribution scores
        if contribution.core_values_alignment > 0.8:
            strengths.append("core_values_alignment")
        elif contribution.core_values_alignment < 0.6:
            improvements.append("core_values_alignment")

        if contribution.architecture_compliance > 0.8:
            strengths.append("architecture_compliance")
        elif contribution.architecture_compliance < 0.6:
            improvements.append("architecture_compliance")

        if contribution.implementation_quality > 0.8:
            strengths.append("implementation_quality")
        elif contribution.implementation_quality < 0.6:
            improvements.append("implementation_quality")

        if contribution.documentation_quality > 0.8:
            strengths.append("documentation_quality")
        elif contribution.documentation_quality < 0.6:
            improvements.append("documentation_quality")

        # Keep only top 3 of each
        return strengths[:3], improvements[:3]

    def _generate_recommendations(self, contributions: List[ContributionRecord],
                                profile: Optional[AgentProfile]) -> List[str]:
        """Generate personalized recommendations for improvement."""
        recommendations = []

        if not profile:
            return ["Continue contributing to build performance profile"]

        # Based on improvement areas
        for area in profile.improvement_areas:
            if area == "core_values_alignment":
                recommendations.append("Review CORE_VALUES.md and focus on 'best part is no part' principle")
            elif area == "architecture_compliance":
                recommendations.append("Study layered architecture and facade pattern in docs/architecture/ARCHITECTURE.md")
            elif area == "implementation_quality":
                recommendations.append("Review existing code patterns and follow established conventions")
            elif area == "documentation_quality":
                recommendations.append("Add comprehensive docstrings and comments for complex logic")

        # Based on contribution types
        if profile.contribution_types.get('refactor', 0) == 0:
            recommendations.append("Try contributing to refactoring tasks to build experience")

        if profile.contribution_types.get('documentation', 0) == 0:
            recommendations.append("Consider documentation contributions to improve knowledge")

        return recommendations[:5]  # Limit to top 5

    def _extract_feedback_themes(self, contributions: List[ContributionRecord]) -> List[str]:
        """Extract common themes from feedback."""
        feedback_texts = [c.review_feedback for c in contributions if c.review_feedback]

        if not feedback_texts:
            return []

        # Simple keyword-based theme extraction
        themes = []
        common_keywords = {
            "simplicity": ["simple", "complex", "over-engineer"],
            "architecture": ["layer", "facade", "domain", "infrastructure"],
            "documentation": ["docstring", "comment", "document"],
            "core_values": ["part", "dependency", "abstraction"],
            "quality": ["test", "bug", "error", "fix"]
        }

        for theme, keywords in common_keywords.items():
            if any(any(keyword in feedback.lower() for keyword in keywords)
                   for feedback in feedback_texts):
                themes.append(theme)

        return themes

    def _suggest_learning_activities(self, contributions: List[ContributionRecord]) -> List[str]:
        """Suggest learning activities based on contribution patterns."""
        suggestions = []

        contribution_types = Counter(c.contribution_type for c in contributions)

        # Suggest based on what's missing
        if contribution_types.get('feature', 0) < contribution_types.get('refactor', 0):
            suggestions.append("Focus on feature development to balance contribution types")

        if not any(c.quality_score > 0.9 for c in contributions):
            suggestions.append("Study high-quality contributions from other agents")

        # Suggest based on quality trends
        recent_scores = [c.quality_score for c in contributions[-5:]]  # Last 5 contributions
        if recent_scores and sum(recent_scores) / len(recent_scores) < 0.7:
            suggestions.append("Review recent feedback and focus on common improvement areas")

        return suggestions

    def _analyze_system_issues(self, contributions: List[ContributionRecord]) -> Dict[str, Any]:
        """Analyze system-wide issues across all agents."""
        if not contributions:
            return {}

        avg_scores = {
            'overall': sum(c.quality_score for c in contributions) / len(contributions),
            'core_values': sum(c.core_values_alignment for c in contributions) / len(contributions),
            'architecture': sum(c.architecture_compliance for c in contributions) / len(contributions),
            'implementation': sum(c.implementation_quality for c in contributions) / len(contributions),
            'documentation': sum(c.documentation_quality for c in contributions) / len(contributions)
        }

        # Find most common issues
        low_score_contributions = [c for c in contributions if c.quality_score < 0.7]
        issue_tags = []
        for contrib in low_score_contributions:
            issue_tags.extend(contrib.tags or [])

        common_issues = Counter(issue_tags).most_common(5)

        return {
            'average_scores': avg_scores,
            'common_issues': dict(common_issues),
            'low_quality_contribution_rate': len(low_score_contributions) / len(contributions)
        }

    def _find_common_weaknesses(self, profiles: Dict[str, AgentProfile]) -> List[str]:
        """Find common improvement areas across all agents."""
        all_improvements = []
        for profile in profiles.values():
            all_improvements.extend(profile.improvement_areas)

        return [area for area, _ in Counter(all_improvements).most_common(3)]

    def _generate_training_recommendations(self, contributions: List[ContributionRecord]) -> List[str]:
        """Generate training recommendations based on contribution analysis."""
        recommendations = []

        system_issues = self._analyze_system_issues(contributions)

        if system_issues.get('average_scores', {}).get('core_values', 1.0) < 0.8:
            recommendations.append("Conduct core values training session for all agents")

        if system_issues.get('average_scores', {}).get('architecture', 1.0) < 0.8:
            recommendations.append("Architecture patterns workshop focusing on layered design")

        common_issues = system_issues.get('common_issues', {})
        if 'documentation' in [issue[0] for issue in common_issues.items() if issue[1] > 5]:
            recommendations.append("Documentation standards training and tooling")

        return recommendations

    def _analyze_quality_distribution(self, contributions: List[ContributionRecord]) -> Dict[str, Any]:
        """Analyze the distribution of quality scores."""
        scores = [c.quality_score for c in contributions]

        if not scores:
            return {}

        return {
            'mean': sum(scores) / len(scores),
            'median': sorted(scores)[len(scores) // 2],
            'high_quality_rate': len([s for s in scores if s >= 0.9]) / len(scores),
            'needs_improvement_rate': len([s for s in scores if s < 0.7]) / len(scores),
            'distribution': {
                'excellent': len([s for s in scores if s >= 0.9]),
                'good': len([s for s in scores if 0.8 <= s < 0.9]),
                'average': len([s for s in scores if 0.7 <= s < 0.8]),
                'needs_work': len([s for s in scores if s < 0.7])
            }
        }


def main():
    """CLI interface for AI feedback system."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ai_feedback_system.py record <agent_id> <type> <quality_score> <files...>")
        print("  python ai_feedback_system.py profile <agent_id>")
        print("  python ai_feedback_system.py feedback <agent_id> [days]")
        print("  python ai_feedback_system.py top_performers [limit]")
        print("  python ai_feedback_system.py training_needs")
        sys.exit(1)

    system = AIFeedbackSystem()
    command = sys.argv[1]

    if command == 'record':
        if len(sys.argv) < 5:
            print("Usage: python ai_feedback_system.py record <agent_id> <type> <quality_score> <files...>")
            sys.exit(1)

        agent_id, contrib_type, quality_score = sys.argv[2], sys.argv[3], float(sys.argv[4])
        files_changed = sys.argv[5:]

        record = ContributionRecord(
            id="",  # Will be set by record_contribution
            agent_id=agent_id,
            timestamp=datetime.now(),
            contribution_type=contrib_type,
            files_changed=files_changed,
            quality_score=quality_score
        )

        contribution_id = system.record_contribution(record)
        print(f"Contribution recorded with ID: {contribution_id}")

    elif command == 'profile':
        if len(sys.argv) < 3:
            print("Usage: python ai_feedback_system.py profile <agent_id>")
            sys.exit(1)

        agent_id = sys.argv[2]
        profile = system.get_agent_profile(agent_id)

        if profile:
            print(json.dumps(asdict(profile), indent=2, default=str))
        else:
            print(f"No profile found for {agent_id}")

    elif command == 'feedback':
        if len(sys.argv) < 3:
            print("Usage: python ai_feedback_system.py feedback <agent_id> [days]")
            sys.exit(1)

        agent_id = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 30

        report = system.generate_feedback_report(agent_id, days)
        print(json.dumps(report, indent=2))

    elif command == 'top_performers':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        top_performers = system.get_top_performers(limit)

        print("Top Performing AI Agents:")
        for i, (agent_id, score) in enumerate(top_performers, 1):
            print(f"{i}. {agent_id}: {score:.3f}")

    elif command == 'training_needs':
        needs = system.identify_training_needs()
        print(json.dumps(needs, indent=2))


if __name__ == '__main__':
    main()

