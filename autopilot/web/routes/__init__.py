"""Route modules for the autopilot-video web console."""

from autopilot.web.routes.dashboard import router as dashboard_router
from autopilot.web.routes.gates import router as gates_router
from autopilot.web.routes.media import router as media_router
from autopilot.web.routes.pipeline import router as pipeline_router
from autopilot.web.routes.review import router as review_router
from autopilot.web.routes.settings import router as settings_router
from autopilot.web.routes.sse import router as sse_router

__all__ = [
    "dashboard_router",
    "gates_router",
    "media_router",
    "pipeline_router",
    "review_router",
    "settings_router",
    "sse_router",
]
