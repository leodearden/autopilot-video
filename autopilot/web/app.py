"""FastAPI application factory for the autopilot-video web console."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

__all__ = ["create_app"]

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def create_app(db_path: str) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to the SQLite catalog database file.

    Returns:
        Configured FastAPI instance with templates, static files, and routes.
    """
    app = FastAPI(title="Autopilot Video Console")

    # Store db_path for lazy CatalogDB connection in route dependencies
    app.state.db_path = db_path

    # Configure Jinja2 templates
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Health check endpoint
    @app.get("/api/health")
    def health_check() -> dict:
        """Return basic health status."""
        return {"status": "ok"}

    return app
