# path: src/juniorhome/dashboard.py
#!/usr/bin/env python3
"""
Terminal Dashboard

Beautiful terminal-based dashboard for JuniorHome.
Shows status, workflows, agents, LLM usage, and system health
in a clean, modern way using the Rich library.
"""

import logging
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from typing import Any, Dict, Optional

from .orchestrator import JuniorHomeOrchestrator

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Dashboard:
    """
    Terminal dashboard for JuniorHome.
    """

    def __init__(self, orchestrator: Optional[JuniorHomeOrchestrator] = None):
        self.orchestrator = orchestrator or JuniorHomeOrchestrator()
        self.console = Console() if HAS_RICH else None

        if not HAS_RICH:
            logging.warning("rich library not installed. Install with: pip install rich")

    def show_status(self):
        if not HAS_RICH:
            print(self.orchestrator.status())
            return

        status = self.orchestrator.status()

        table = Table(title="JuniorHome Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for key, value in status.items():
            if isinstance(value, bool):
                value = "✓" if value else "✗"
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)

    def show_workflows(self):
        if not HAS_RICH:
            print("Workflows:", self.orchestrator.workflow_engine.list_workflows())
            return

        workflows = self.orchestrator.workflow_engine.list_workflows()

        table = Table(title="Registered Workflows")
        table.add_column("Workflow Name", style="cyan")

        for wf in workflows:
            table.add_row(wf)

        self.console.print(table)

    def show_full_dashboard(self):
        if not HAS_RICH:
            print(self.orchestrator.status())
            return

        layout = Layout()
        layout.split_column(
            Layout(Panel("JuniorHome Dashboard", style="bold blue"), size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(self._status_panel(), name="status"),
            Layout(self._workflows_panel(), name="workflows")
        )

        self.console.print(layout)

    def _status_panel(self):
        status = self.orchestrator.status()
        table = Table.grid(padding=1)
        for key, value in status.items():
            if isinstance(value, bool):
                value = "✓" if value else "✗"
            table.add_row(f"[cyan]{key.replace('_', ' ').title()}[/cyan]:", str(value))
        return Panel(table, title="System Status", border_style="green")

    def _workflows_panel(self):
        workflows = self.orchestrator.workflow_engine.list_workflows()
        table = Table.grid(padding=1)
        for wf in workflows:
            table.add_row(f"[magenta]•[/magenta] {wf}")
        return Panel(table, title="Workflows", border_style="blue")
