"""Shared route-handler dependencies for the web layer."""

from __future__ import annotations

from fastapi import Request

from autopilot.db import CatalogDB


def get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def is_htmx(request: Request) -> bool:
    """Check if the request is from HTMX."""
    return request.headers.get("hx-request") == "true"
