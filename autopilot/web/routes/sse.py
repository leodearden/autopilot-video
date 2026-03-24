"""Server-Sent Events endpoint for real-time pipeline event streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse, ServerSentEvent  # type: ignore[attr-defined]

from autopilot.web.deps import get_db

logger = logging.getLogger(__name__)

_PRUNE_INTERVAL = 60  # prune every ~60 poll cycles (~1 minute)

VALID_EVENT_TYPES: tuple[str, ...] = (
    "stage_started",
    "stage_completed",
    "stage_error",
    "job_started",
    "job_completed",
    "job_error",
    "job_progress",
    "gate_waiting",
    "gate_approved",
    "gate_skipped",
    "run_completed",
    "run_failed",
)

router = APIRouter()


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


def _get_last_event_id(request: Request) -> int:
    """Parse Last-Event-ID header, returning 0 on missing/invalid."""
    raw = request.headers.get("last-event-id", "0")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


async def _event_generator(request: Request):
    """Async generator that polls pipeline_events and yields SSE events."""
    db = get_db(request)
    last_event_id = _get_last_event_id(request)
    poll_count = 0
    try:
        while True:
            events = db.get_events_since(last_event_id)
            for ev in events:
                yield _format_event(ev)
                last_event_id = int(ev["event_id"])  # type: ignore[call-overload]

            poll_count += 1
            if poll_count % _PRUNE_INTERVAL == 0:
                try:
                    db.prune_events(hours=24)
                except Exception:
                    logger.warning("Failed to prune events", exc_info=True)

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
