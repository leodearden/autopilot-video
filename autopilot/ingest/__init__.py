"""Ingest pipeline — media scanning, audio normalization, and deduplication."""

from autopilot.ingest.dedup import compute_hash, find_duplicates, mark_duplicates
from autopilot.ingest.normalizer import normalize_audio
from autopilot.ingest.scanner import MediaFile, probe_file, scan_directory

__all__ = [
    "MediaFile",
    "compute_hash",
    "find_duplicates",
    "mark_duplicates",
    "normalize_audio",
    "probe_file",
    "scan_directory",
]
