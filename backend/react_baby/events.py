"""WebSocket fan-out for progress, logs and job lifecycle messages."""

from __future__ import annotations

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        # Copy: a failed send removes the socket while we're iterating.
        for connection in list(self.active):
            try:
                await connection.send_json(message)
            except Exception:
                self.active.discard(connection)


manager = ConnectionManager()
