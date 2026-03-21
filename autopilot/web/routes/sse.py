"""Server-Sent Events endpoint for real-time pipeline event streaming."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from autopilot.db import CatalogDB

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def _format_event(event: dict) -> ServerSentEvent:
    """Format a pipeline_events row as a ServerSentEvent.

    Merges parsed payload_json into the data dict alongside
    event_type, stage, and job_id for flat JSON consumption.
    """
    data: dict = {
        "event_type": event["event_type"],
        "stage": event.get("stage"),
        "job_id": event.get("job_id"),
    }
    payload_json = event.get("payload_json")
    if payload_json:
        try:
            data.update(json.loads(payload_json))
        except (json.JSONDecodeError, TypeError):
            data["payload_json"] = payload_json

    return ServerSentEvent(
        id=str(event["event_id"]),
        event=event["event_type"],
        data=json.dumps(data),
    )


async def _event_generator(request: Request):
    """Async generator that polls pipeline_events and yields SSE events."""
    db = _get_db(request)
    last_event_id = 0
    try:
        while True:
            events = db.get_events_since(last_event_id)
            for ev in events:
                yield _format_event(ev)
                last_event_id = ev["event_id"]
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        return
    finally:
        db.close()


@router.get("/api/events")
async def sse_events(request: Request):
    """Stream pipeline events as Server-Sent Events."""
    return EventSourceResponse(
        _event_generator(request),
        ping=15,
    )
