# path: src/juniorhome/websocket_server.py
#!/usr/bin/env python3
"""
WebSocket Server

Optional WebSocket server for real-time updates from JuniorHome.
Useful for live dashboards, agent monitoring, pipeline status, and visualization updates.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    import asyncio
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class WebSocketServer:
    """
    Simple WebSocket server for broadcasting real-time events.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: List[Any] = []
        self.event_handlers: Dict[str, List[Callable]] = {}

        if not HAS_WEBSOCKETS:
            logging.warning("websockets library not installed. WebSocket server disabled.")

    async def _handler(self, websocket, path):
        self.clients.append(websocket)
        logging.info(f"New WebSocket client connected. Total: {len(self.clients)}")

        try:
            async for message in websocket:
                # Handle incoming messages if needed
                logging.debug(f"Received message: {message}")
        except Exception as e:
            logging.warning(f"WebSocket client error: {e}")
        finally:
            self.clients.remove(websocket)
            logging.info(f"WebSocket client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message: str):
        if not self.clients:
            return

        disconnected = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self.clients.remove(client)

    def start(self):
        if not HAS_WEBSOCKETS:
            print("websockets is required. Install with: pip install websockets")
            return

        import asyncio

        async def main():
            async with websockets.serve(self._handler, self.host, self.port):
                logging.info(f"WebSocket server started on ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        asyncio.run(main())

    def publish_event(self, event_type: str, data: Any):
        import json
        message = json.dumps({"type": event_type, "data": data})
        # In a full implementation, this would be called from an async context
        logging.info(f"Event published: {event_type}")
