"""End-to-end integration tests for the full web console.

These tests verify cross-cutting workflows and data consistency across
multiple web console views. Individual endpoint tests live in their
respective test files (test_dashboard.py, test_gates.py, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app

PIPELINE_STAGES = (
    "ingest", "analyze", "classify", "narrate", "script",
    "edl", "source", "render", "upload",
)
