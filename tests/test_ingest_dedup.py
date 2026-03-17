"""Tests for autopilot.ingest.dedup — SHA-256 hashing and duplicate detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
