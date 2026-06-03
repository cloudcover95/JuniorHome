# path: src/juniorhome/enhanced_dashboard.py
#!/usr/bin/env python3
"""
Enhanced Dashboard

Extended dashboard with support for real-time updates via WebSocket
and integration with ObservabilityManager.
"""

import logging
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .dashboard import Dashboard
from .observability_manager import ObservabilityManager

from .websocket_server import WebSocketServer

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EnhancedDashboard(Dashboard):
    """
    Dashboard with real-time update capabilities.
    """

    def __init__(self, orchestrator: Optional[Any] = None, observability: Optional[ObservabilityManager] = None):
        super().__init__(orchestrator)
        self.observability = observability or ObservabilityManager()
        self.websocket = None

    def enable_realtime(self, host: str = "0.0.0.0", port: int = 8765):
        if not HAS_RICH:
            logging.warning("Rich not available for enhanced dashboard")
            return

        self.websocket = WebSocketServer(host=host, port=port)
        logging.info("Real-time updates enabled")

    def show_live_status(self, refresh_rate: float = 2.0):
        if not HAS_RICH:
            print(self.orchestrator.status())
            return

        console = Console()

        def generate_layout():
            status = self.orchestrator.status()
            health = self.observability.run_health_checks()

            status_table = Table.grid(padding=1)
            for key, value in status.items():
                if isinstance(value, bool):
                    value = "✓" if value else "✗"
                status_table.add_row(f"[cyan]{key.replace('_', ' ').title()}[/cyan]:", str(value))

            health_table = Table.grid(padding=1)
            for check in health.get("checks", []):
                status_icon = "✓" if check.get("status") == "healthy" else "✗"
                health_table.add_row(f"[green]{check['name']}[/green]:", status_icon)

            layout = Panel(
                Table.grid().add_row(
                    Panel(status_table, title="System Status", border_style="green"),
                    Panel(health_table, title="Health Checks", border_style="blue"),
                ),
                title="JuniorHome Live Dashboard",
                border_style="magenta",
            )
            return layout

        with Live(generate_layout(), refresh_per_second=1 / refresh_rate, console=console) as live:
            try:
                while True:
                    live.update(generate_layout())
            except KeyboardInterrupt:
                pass
