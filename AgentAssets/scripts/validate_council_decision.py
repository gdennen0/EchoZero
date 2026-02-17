#!/usr/bin/env python3
"""
Council decision validation script.

Validates that council decisions follow the required format and include
all necessary analysis from each council member.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class CouncilDecisionValidator:
    """Validates council decision documents."""

    REQUIRED_ROLES = {'architect', 'systems', 'ux', 'pragmatic'}
    VALID_VOTES = {'approve', 'approve_with_conditions', 'reject_with_alternative', 'reject'}

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_decision_file(self, file_path: str) -> bool:
        """Validate a council decision file."""
        self.errors = []
        self.warnings = []

        try:
            with open(file_path, 'r') as f:
                decision = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.errors.append(f"Failed to load decision file: {e}")
            return False

        return self.validate_decision_structure(decision)

    def validate_decision_structure(self, decision: Dict[str, Any]) -> bool:
        """Validate the structure of a council decision."""

        # Check required top-level fields
        required_fields = [
            'proposal_type', 'proposal_title', 'proposal_description',
            'council_analyses', 'recommendation'
        ]

        for field in required_fields:
            if field not in decision:
                self.errors.append(f"Missing required field: {field}")
                return False

        # Validate proposal type
        valid_types = {'feature', 'refactor', 'bug_fix', 'architectural', 'documentation'}
        if decision['proposal_type'] not in valid_types:
            self.errors.append(f"Invalid proposal_type: {decision['proposal_type']}. Must be one of {valid_types}")
            return False

        # Validate council analyses
        if not isinstance(decision['council_analyses'], list):
            self.errors.append("council_analyses must be a list")
            return False

        if len(decision['council_analyses']) != 4:
            self.errors.append(f"Expected 4 council analyses, got {len(decision['council_analyses'])}")
            return False

        # Check all required roles are present
        roles_present = {analysis.get('role') for analysis in decision['council_analyses']}
        if roles_present != self.REQUIRED_ROLES:
            missing = self.REQUIRED_ROLES - roles_present
            extra = roles_present - self.REQUIRED_ROLES
            if missing:
                self.errors.append(f"Missing council roles: {missing}")
            if extra:
                self.errors.append(f"Extra council roles: {extra}")
            return False

        # Validate each analysis
        for i, analysis in enumerate(decision['council_analyses']):
            if not self._validate_council_analysis(analysis, i):
                return False

        # Validate recommendation
        if not self._validate_recommendation(decision.get('recommendation', '')):
            return False

        # Check for implementation status if present
        if 'implementation_status' in decision:
            valid_statuses = {'not_started', 'in_progress', 'completed', 'cancelled'}
            if decision['implementation_status'] not in valid_statuses:
                self.errors.append(f"Invalid implementation_status: {decision['implementation_status']}")
                return False

        return len(self.errors) == 0

    def _validate_council_analysis(self, analysis: Dict[str, Any], index: int) -> bool:
        """Validate a single council analysis."""
        required_fields = ['role', 'vote', 'concerns', 'rationale']

        for field in required_fields:
            if field not in analysis:
                self.errors.append(f"Analysis {index}: Missing required field '{field}'")
                return False

        # Validate role
        if analysis['role'] not in self.REQUIRED_ROLES:
            self.errors.append(f"Analysis {index}: Invalid role '{analysis['role']}'")
            return False

        # Validate vote
        if analysis['vote'] not in self.VALID_VOTES:
            self.errors.append(f"Analysis {index}: Invalid vote '{analysis['vote']}'. Must be one of {self.VALID_VOTES}")
            return False

        # Validate concerns
        if not isinstance(analysis['concerns'], list):
            self.errors.append(f"Analysis {index}: 'concerns' must be a list")
            return False

        if len(analysis['concerns']) == 0:
            self.warnings.append(f"Analysis {index}: No concerns listed - consider if analysis is thorough enough")

        # Validate rationale
        if not isinstance(analysis['rationale'], str) or len(analysis['rationale']) < 10:
            self.errors.append(f"Analysis {index}: 'rationale' must be a non-empty string with meaningful content")
            return False

        # Check for conditions/alternative if vote requires it
        if analysis['vote'] == 'approve_with_conditions' and 'conditions_or_alternative' not in analysis:
            self.errors.append(f"Analysis {index}: 'approve_with_conditions' vote requires 'conditions_or_alternative' field")
            return False

        if analysis['vote'] == 'reject_with_alternative' and 'conditions_or_alternative' not in analysis:
            self.errors.append(f"Analysis {index}: 'reject_with_alternative' vote requires 'conditions_or_alternative' field")
            return False

        return True

    def _validate_recommendation(self, recommendation: str) -> bool:
        """Validate the final recommendation."""
        if not isinstance(recommendation, str) or len(recommendation.strip()) == 0:
            self.errors.append("Recommendation must be a non-empty string")
            return False

        # Check for proper format (should start with RECOMMENDATION:)
        if not recommendation.strip().startswith('RECOMMENDATION:'):
            self.warnings.append("Recommendation should start with 'RECOMMENDATION:' for consistency")

        # Check for unanimous language
        if 'unanimous' not in recommendation.lower() and 'unanimously' not in recommendation.lower():
            self.warnings.append("Consider using 'unanimous' or 'unanimously' to indicate council agreement")

        return True

    def get_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


def main():
    """CLI interface for council decision validation."""
    if len(sys.argv) != 2:
        print("Usage: python validate_council_decision.py <decision_file.json>")
        sys.exit(1)

    decision_file = sys.argv[1]

    validator = CouncilDecisionValidator()
    is_valid = validator.validate_decision_file(decision_file)
    report = validator.get_report()

    print(f"Validation Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"Errors: {report['error_count']}, Warnings: {report['warning_count']}")

    if report['errors']:
        print("\n❌ ERRORS:")
        for error in report['errors']:
            print(f"  - {error}")

    if report['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")

    if is_valid and report['warnings']:
        print("\n✅ Decision is valid but consider addressing warnings for better quality.")

    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()

