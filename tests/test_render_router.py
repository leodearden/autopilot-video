"""Tests for autopilot.render.router — render routing and assembly."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_routing_error_importable_and_is_exception(self) -> None:
        """RoutingError should be importable and be an Exception subclass."""
        from autopilot.render.router import RoutingError

        assert issubclass(RoutingError, Exception)

    def test_route_and_render_importable(self) -> None:
        """route_and_render should be importable and callable."""
        from autopilot.render.router import route_and_render

        assert callable(route_and_render)

    def test_route_and_render_signature(self) -> None:
        """route_and_render should accept (narrative_id, db, config)."""
        from autopilot.render.router import route_and_render

        sig = inspect.signature(route_and_render)
        param_names = list(sig.parameters.keys())
        assert param_names == ["narrative_id", "db", "config"]

    def test_all_exports(self) -> None:
        """__all__ should export RoutingError and route_and_render."""
        from autopilot.render import router

        assert hasattr(router, "__all__")
        assert set(router.__all__) == {"RoutingError", "route_and_render"}
