# path: examples/basic_orchestration.py
#!/usr/bin/env python3
"""
Basic Integration Example for JuniorHome

Demonstrates wiring DataLake + Reporter + Orchestrator.
"""

from juniorhome import (
    JuniorHomeOrchestrator,
    DataLake,
    Reporter,
)


def main():
    print("=== JuniorHome Basic Orchestration Example ===\n")

    # Initialize orchestrator
    orchestrator = JuniorHomeOrchestrator()

    # Example: Generate an intelligent report
    report = orchestrator.generate_intelligent_report("market_analysis")
    print("Generated Report:", report)

    print("\nOrchestrator Status:", orchestrator.status())


if __name__ == "__main__":
    main()
