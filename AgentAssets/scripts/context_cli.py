#!/usr/bin/env python3
"""
AgentAssets Context CLI for AI Agents.

Command-line interface for AI agents to get intelligent context
from the AgentAssets learning system.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any


class ContextCLI:
    """Command-line interface for context access."""

    def __init__(self):
        # Import context provider
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from context_provider import AgentAssetsContextProvider, ContextRequest
            self.context_provider = AgentAssetsContextProvider()
            self.ContextRequest = ContextRequest
        except ImportError as e:
            print(f"Error: {e}")
            print("Make sure you're in the AgentAssets/scripts directory")
            sys.exit(1)

    def get_context(self, agent_id: str, query: str, **kwargs) -> Dict[str, Any]:
        """Get context for an agent."""
        request = self.ContextRequest(
            agent_id=agent_id,
            query=query,
            context_type=kwargs.get('context_type', 'general'),
            domain=kwargs.get('domain'),
            current_task=kwargs.get('task'),
            metadata=kwargs
        )

        response = self.context_provider.get_context_for_agent(request)
        return self._format_response(response)

    def get_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get insights for an agent."""
        return self.context_provider.get_agent_insights(agent_id)

    def submit_feedback(self, agent_id: str, **feedback) -> bool:
        """Submit feedback for an agent."""
        return self.context_provider.update_agent_feedback(agent_id, feedback)

    def list_agents(self) -> Dict[str, Any]:
        """List all known agents."""
        agents = list(self.context_provider.agent_profiles.keys())
        return {"agents": agents, "count": len(agents)}

    def query(self, query: str, agent_id: str = "cli_user") -> Dict[str, Any]:
        """Query the system with natural language."""
        if self.context_provider.learning_engine:
            return self.context_provider.learning_engine.get_context_for_query(
                query, {"agent_id": agent_id}
            )
        return {"error": "Query system not available"}

    def _format_response(self, response) -> Dict[str, Any]:
        """Format response for CLI output."""
        return {
            "request_id": response.request_id,
            "relevance_score": response.relevance_score,
            "core_context": response.core_context,
            "agent_specific_context": response.agent_specific_context,
            "learning_insights": response.learning_insights,
            "generated_at": response.generated_at.isoformat()
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentAssets Context CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get context for creating a block
  python context_cli.py context claude-3 "How do I create a new audio block?"

  # Get agent insights
  python context_cli.py insights claude-3

  # Submit feedback
  python context_cli.py feedback claude-3 --helpful 0.9 --preferred_types "patterns,examples"

  # List all agents
  python context_cli.py agents

  # Natural language query
  python context_cli.py query "What are the core values?" --agent-id gpt-4

Usage in AI Chat Interfaces:
  Run: python context_cli.py context <your-agent-id> "<your question>"
  Copy the JSON output into your response
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Context command
    context_parser = subparsers.add_parser('context', help='Get context for a query')
    context_parser.add_argument('agent_id', help='Your AI agent identifier (e.g., claude-3, gpt-4)')
    context_parser.add_argument('query', help='Your question or task description')
    context_parser.add_argument('--type', dest='context_type', default='general',
                               choices=['general', 'specific', 'task', 'domain'],
                               help='Type of context requested')
    context_parser.add_argument('--domain', help='Specific domain (e.g., audio, ui, blocks)')
    context_parser.add_argument('--task', help='Current task being worked on')

    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Get insights about an agent')
    insights_parser.add_argument('agent_id', help='Agent identifier')

    # Feedback command
    feedback_parser = subparsers.add_parser('feedback', help='Submit feedback')
    feedback_parser.add_argument('agent_id', help='Agent identifier')
    feedback_parser.add_argument('--helpful', type=float, help='How helpful was the context? (0.0-1.0)')
    feedback_parser.add_argument('--preferred-types', help='Comma-separated preferred context types')
    feedback_parser.add_argument('--successful-patterns', help='Comma-separated successful patterns used')

    # Agents command
    subparsers.add_parser('agents', help='List all known agents')

    # Query command
    query_parser = subparsers.add_parser('query', help='Natural language query')
    query_parser.add_argument('query', help='Natural language query')
    query_parser.add_argument('--agent-id', default='cli_user', help='Agent identifier')

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    # Initialize CLI
    cli = ContextCLI()

    try:
        if args.command == 'context':
            # Build kwargs
            kwargs = {
                'context_type': args.context_type
            }
            if args.domain:
                kwargs['domain'] = args.domain
            if args.task:
                kwargs['task'] = args.task

            result = cli.get_context(args.agent_id, args.query, **kwargs)

        elif args.command == 'insights':
            result = cli.get_insights(args.agent_id)

        elif args.command == 'feedback':
            feedback = {}
            if args.helpful is not None:
                feedback['context_helpful'] = args.helpful
            if args.preferred_types:
                feedback['preferred_context_types'] = {
                    t.strip(): 1.0 for t in args.preferred_types.split(',')
                }
            if args.successful_patterns:
                feedback['successful_patterns'] = [
                    p.strip() for p in args.successful_patterns.split(',')
                ]

            success = cli.submit_feedback(args.agent_id, **feedback)
            result = {"success": success, "feedback_submitted": bool(feedback)}

        elif args.command == 'agents':
            result = cli.list_agents()

        elif args.command == 'query':
            result = cli.query(args.query, args.agent_id)

        else:
            parser.print_help()
            return

        # Output result as JSON
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

