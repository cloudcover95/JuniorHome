# path: src/juniorhome/cli.py
#!/usr/bin/env python3
"""
Command Line Interface

Provides a user-friendly CLI for interacting with JuniorHome.
Uses argparse for simplicity and broad compatibility.
"""

import argparse
import sys

from .application import Application
from .orchestrator import JuniorHomeOrchestrator

from .dashboard import Dashboard


def main():
    parser = argparse.ArgumentParser(description="JuniorHome - Sovereign Edge Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--config", help="Path to config file")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show interactive dashboard")
    dashboard_parser.add_argument("--config", help="Path to config file")

    # Process vault command
    process_parser = subparsers.add_parser("process-vault", help="Process Obsidian vault once")
    process_parser.add_argument("vault", help="Path to Obsidian vault")
    process_parser.add_argument("--config", help="Path to config file")

    # Run workflow command
    workflow_parser = subparsers.add_parser("run-workflow", help="Run a registered workflow")
    workflow_parser.add_argument("name", help="Workflow name")
    workflow_parser.add_argument("--config", help="Path to config file")

    # LLM query command
    llm_parser = subparsers.add_parser("llm", help="Query the LLM router")
    llm_parser.add_argument("prompt", help="Prompt to send")
    llm_parser.add_argument("--bitnet", action="store_true", help="Prefer BitNet-mlx backend")
    llm_parser.add_argument("--model", default="llama3.2", help="Ollama model to use")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    app = Application(config_file=getattr(args, "config", None))

    if args.command == "status":
        print(app.get_status())

    elif args.command == "dashboard":
        dashboard = Dashboard(orchestrator=app.orchestrator)
        dashboard.show_full_dashboard()

    elif args.command == "process-vault":
        from .knowledge_service import KnowledgeService
        service = KnowledgeService(vault_path=args.vault, enable_scheduling=False)
        results = service.process_vault_once()
        print(f"Processed {len(results)} files")

    elif args.command == "run-workflow":
        results = app.orchestrator.workflow_engine.run(args.name)
        print(results)

    elif args.command == "llm":
        result = app.orchestrator.route_llm(
            prompt=args.prompt,
            prefer_bitnet=args.bitnet,
            model=args.model,
        )
        print(result.get("response", result))


if __name__ == "__main__":
    main()
