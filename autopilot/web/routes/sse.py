"""Server-Sent Events endpoint for real-time pipeline event streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from autopilot.db import CatalogDB

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


async def _event_generator(request: Request):
    """Async generator that polls pipeline_events and yields SSE events."""
    try:
        while True:
            await asyncio.sleep(1)
            yield {}
    except asyncio.CancelledError:
        return


@router.get("/api/events")
async def sse_events(request: Request):
    """Stream pipeline events as Server-Sent Events."""
    return EventSourceResponse(
        _event_generator(request),
        ping=15,
    )
