"""OTIO export: convert EDL structures to OpenTimelineIO files.

Provides export_otio() for converting JSON EDL dicts (from edl.py) to
.otio files for NLE review, and detect_otio_changes() for round-trip
change detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

__all__ = [
    "OtioExportError",
    "export_otio",
    "detect_otio_changes",
]


class OtioExportError(Exception):
    """Raised for all OTIO export failures."""


def export_otio(edl: dict, output_path: Path, db: CatalogDB) -> Path:
    """Export an EDL structure to an OpenTimelineIO .otio file.

    Args:
        edl: EDL dict with clips, transitions, crop_modes, etc.
        output_path: Destination path for the .otio file.
        db: CatalogDB for media file lookups.

    Returns:
        The output_path on success.

    Raises:
        OtioExportError: If export fails.
    """
    raise NotImplementedError


def detect_otio_changes(otio_path: Path, original_edl: dict) -> dict:
    """Detect whether an .otio file has been modified from the original EDL.

    Args:
        otio_path: Path to the .otio file to check.
        original_edl: The original EDL dict used to generate the .otio file.

    Returns:
        Dict with 'modified' bool and 'changes' list of descriptions.

    Raises:
        OtioExportError: If the .otio file cannot be read.
    """
    raise NotImplementedError
