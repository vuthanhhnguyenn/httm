from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any


class WebSocketTestClient:
    """Minimal async WebSocket test double for envelope and reconnect tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("websocket is closed")
        self.sent.append(dict(payload))

    async def receive_json(self) -> dict[str, Any]:
        return await self._incoming.get()

    async def feed_json(self, payload: Mapping[str, Any]) -> None:
        await self._incoming.put(dict(payload))

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        self.close_code = code

