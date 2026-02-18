#!/usr/bin/env python3
"""
AgentAssets Context API for AI Agents.

Provides a simple HTTP API that AI agents can call to get intelligent context
based on their learning profiles and the system's understanding.
"""

import json
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from typing import Dict, Any
import threading
import time


class ContextAPI:
    """HTTP API for providing context to AI agents."""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web access

        # Import context provider
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from context_provider import AgentAssetsContextProvider
            self.context_provider = AgentAssetsContextProvider()
        except ImportError as e:
            print(f"Error importing context provider: {e}")
            self.context_provider = None

        self.setup_routes()

    def setup_routes(self):
        """Set up API routes."""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "provider_available": self.context_provider is not None
            })

        @self.app.route('/context', methods=['POST'])
        def get_context():
            """Get context for an AI agent."""
            try:
                data = request.get_json()

                if not data or 'agent_id' not in data:
                    return jsonify({"error": "agent_id is required"}), 400

                # Create context request
                from context_provider import ContextRequest

                context_request = ContextRequest(
                    agent_id=data['agent_id'],
                    query=data.get('query', ''),
                    context_type=data.get('context_type', 'general'),
                    domain=data.get('domain'),
                    current_task=data.get('current_task'),
                    conversation_history=data.get('conversation_history', []),
                    metadata=data.get('metadata', {})
                )

                # Get context
                if self.context_provider:
                    response = self.context_provider.get_context_for_agent(context_request)

                    # Convert to JSON-serializable format
                    response_data = {
                        "request_id": response.request_id,
                        "agent_id": response.agent_id,
                        "core_context": response.core_context,
                        "agent_specific_context": response.agent_specific_context,
                        "learning_insights": response.learning_insights,
                        "relevance_score": response.relevance_score,
                        "generated_at": response.generated_at.isoformat()
                    }

                    return jsonify(response_data)
                else:
                    return jsonify({"error": "Context provider not available"}), 503

            except Exception as e:
                return jsonify({"error": f"Internal error: {str(e)}"}), 500

        @self.app.route('/insights/<agent_id>', methods=['GET'])
        def get_insights(agent_id: str):
            """Get insights about a specific agent."""
            try:
                if self.context_provider:
                    insights = self.context_provider.get_agent_insights(agent_id)
                    return jsonify(insights)
                else:
                    return jsonify({"error": "Context provider not available"}), 503

            except Exception as e:
                return jsonify({"error": f"Internal error: {str(e)}"}), 500

        @self.app.route('/feedback', methods=['POST'])
        def submit_feedback():
            """Submit feedback about provided context."""
            try:
                data = request.get_json()

                if not data or 'agent_id' not in data or 'feedback' not in data:
                    return jsonify({"error": "agent_id and feedback are required"}), 400

                if self.context_provider:
                    success = self.context_provider.update_agent_feedback(
                        data['agent_id'],
                        data['feedback']
                    )

                    return jsonify({"success": success})
                else:
                    return jsonify({"error": "Context provider not available"}), 503

            except Exception as e:
                return jsonify({"error": f"Internal error: {str(e)}"}), 500

        @self.app.route('/agents', methods=['GET'])
        def list_agents():
            """List all known agents."""
            try:
                if self.context_provider:
                    agents = list(self.context_provider.agent_profiles.keys())
                    return jsonify({"agents": agents})
                else:
                    return jsonify({"error": "Context provider not available"}), 503

            except Exception as e:
                return jsonify({"error": f"Internal error: {str(e)}"}), 500

        @self.app.route('/query', methods=['POST'])
        def query_context():
            """Query context with natural language."""
            try:
                data = request.get_json()

                if not data or 'query' not in data:
                    return jsonify({"error": "query is required"}), 400

                agent_id = data.get('agent_id', 'anonymous')
                query = data['query']

                # Use learning engine for query analysis
                if self.context_provider and self.context_provider.learning_engine:
                    context = self.context_provider.learning_engine.get_context_for_query(
                        query,
                        {"agent_id": agent_id}
                    )

                    return jsonify(context)
                else:
                    return jsonify({"error": "Query system not available"}), 503

            except Exception as e:
                return jsonify({"error": f"Internal error: {str(e)}"}), 500

    def run(self, host: str = 'localhost', port: int = 5001, debug: bool = False):
        """Run the API server."""
        print(f"🚀 Starting AgentAssets Context API on {host}:{port}")
        print("Available endpoints:")
        print("  GET  /health          - Health check")
        print("  POST /context         - Get context for agent")
        print("  GET  /insights/<id>   - Get agent insights")
        print("  POST /feedback        - Submit feedback")
        print("  GET  /agents          - List known agents")
        print("  POST /query           - Natural language query")
        print("\nExample usage in AI chat:")
        print('curl -X POST http://localhost:5001/context \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"agent_id": "claude-3", "query": "How do I create a new block?"}\'')

        self.app.run(host=host, port=port, debug=debug)


def main():
    """Run the context API server."""
    import argparse

    parser = argparse.ArgumentParser(description='AgentAssets Context API')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    api = ContextAPI()
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

