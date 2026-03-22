"""Route modules for the autopilot-video web console."""

from autopilot.web.routes.media import router as media_router
from autopilot.web.routes.sse import router as sse_router

__all__ = ["media_router", "sse_router"]
