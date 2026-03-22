"""Media list page and API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from autopilot.db import CatalogDB

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


@router.get("/api/media")
def api_media(
    request: Request,
    q: str | None = Query(None),
    status: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Return paginated, filtered media list as JSON."""
    db = _get_db(request)
    try:
        result = db.query_media(
            q=q,
            status=status,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            order=order,
            page=page,
            per_page=per_page,
        )
    finally:
        db.close()
    return result
