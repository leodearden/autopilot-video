"""Route modules for the autopilot-video web console."""

from autopilot.web.routes.dashboard import router as dashboard_router
from autopilot.web.routes.gates import router as gates_router
from autopilot.web.routes.media import router as media_router
from autopilot.web.routes.sse import router as sse_router

__all__ = ["dashboard_router", "gates_router", "media_router", "sse_router"]
