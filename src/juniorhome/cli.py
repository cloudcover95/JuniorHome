# path: src/juniorhome/cli.py
#!/usr/bin/env python3
"""
JuniorHome Command Line Interface

Simple CLI for interacting with the orchestrator.
"""

import argparse
import sys

from .orchestrator import JuniorHomeOrchestrator


def main():
    parser = argparse.ArgumentParser(description="JuniorHome Orchestrator CLI")
    parser.add_argument("--config", help="Path to config file", default=None)
    parser.add_argument("report", nargs="?", help="Generate a report for a topic")
    args = parser.parse_args()

    orchestrator = JuniorHomeOrchestrator(config_path=args.config)

    if args.report:
        report = orchestrator.generate_intelligent_report(args.report)
        print(report)
    else:
        print("JuniorHome Orchestrator")
        print("Status:", orchestrator.status())


if __name__ == "__main__":
    main()
