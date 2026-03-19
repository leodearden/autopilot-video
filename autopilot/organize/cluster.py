"""Activity clustering using temporal-spatial DBSCAN and semantic refinement.

Provides cluster_activities() for grouping media files into activity clusters
based on timestamps, GPS coordinates, and SigLIP embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

__all__ = ["ActivityCluster", "ClusterError", "cluster_activities"]

logger = logging.getLogger(__name__)


class ClusterError(Exception):
    """Raised for all activity clustering failures."""


@dataclass
class ActivityCluster:
    """Represents a group of temporally/spatially co-located media clips."""

    cluster_id: str
    clip_ids: list[str]
    time_start: str
    time_end: str
    gps_center_lat: float | None = None
    gps_center_lon: float | None = None
    label: str | None = None
    description: str | None = None


# -- Constants -----------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000  # Earth radius in meters
_TEMPORAL_THRESHOLD_S = 30 * 60  # 30 minutes in seconds
_SPATIAL_THRESHOLD_M = 500  # 500 meters


# -- Helpers -------------------------------------------------------------------


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in meters between two lat/lon points.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in meters.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_M * c


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a datetime."""
    # Handle both with and without timezone
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Fallback: try replacing Z with +00:00
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _temporal_spatial_cluster(
    clips: list[dict[str, object]],
    *,
    temporal_threshold_s: float = _TEMPORAL_THRESHOLD_S,
    spatial_threshold_m: float = _SPATIAL_THRESHOLD_M,
) -> list[list[str]]:
    """Cluster clips by temporal and spatial proximity using DBSCAN.

    Args:
        clips: List of media file dicts with id, created_at, gps_lat, gps_lon.
        temporal_threshold_s: Maximum seconds between clips in same cluster.
        spatial_threshold_m: Maximum meters between clips in same cluster.

    Returns:
        List of clusters, each a list of clip IDs.
    """
    if not clips:
        return []

    import numpy as np  # type: ignore[reportMissingImports]
    from sklearn.cluster import DBSCAN  # type: ignore[reportMissingImports]

    n = len(clips)

    # Parse timestamps
    timestamps = []
    for clip in clips:
        ts_str = str(clip["created_at"])
        dt = _parse_iso(ts_str)
        timestamps.append(dt.timestamp())

    # Build precomputed distance matrix
    # d(i,j) = max(temporal_norm, spatial_norm) where each is 0-1 scaled
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            # Temporal distance normalized by threshold
            t_diff = abs(timestamps[i] - timestamps[j])
            t_norm = t_diff / temporal_threshold_s

            # Spatial distance
            lat_i = clips[i].get("gps_lat")
            lon_i = clips[i].get("gps_lon")
            lat_j = clips[j].get("gps_lat")
            lon_j = clips[j].get("gps_lon")

            if (
                lat_i is not None
                and lon_i is not None
                and lat_j is not None
                and lon_j is not None
            ):
                s_dist = _haversine(
                    float(lat_i), float(lon_i), float(lat_j), float(lon_j)
                )
                s_norm = s_dist / spatial_threshold_m
                # Combined: both must be within threshold (max of normalized)
                combined = max(t_norm, s_norm)
            else:
                # Missing GPS: use temporal-only
                combined = t_norm

            dist_matrix[i, j] = combined
            dist_matrix[j, i] = combined

    # DBSCAN with eps=1.0 (since distances are normalized to threshold)
    db = DBSCAN(eps=1.0, min_samples=1, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    # Group clips by label
    cluster_map: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        clip_id = str(clips[idx]["id"])
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(clip_id)

    return list(cluster_map.values())


def cluster_activities(db: CatalogDB) -> list[ActivityCluster]:
    """Cluster all media files into activity groups.

    Loads media files from the database, runs temporal-spatial DBSCAN clustering,
    optionally refines with semantic embeddings, and stores results.

    Args:
        db: Catalog database instance.

    Returns:
        List of ActivityCluster instances.
    """
    # Load all media
    media = db.list_all_media()
    if not media:
        logger.info("No media files found, nothing to cluster")
        return []

    # Filter to only those with timestamps
    clips = [m for m in media if m.get("created_at") is not None]
    if not clips:
        logger.warning("No media files with timestamps, cannot cluster")
        return []

    logger.info("Clustering %d media files", len(clips))

    # Clear existing clusters for idempotency
    db.clear_activity_clusters()

    # Phase 1: Temporal-spatial clustering
    raw_clusters = _temporal_spatial_cluster(clips)

    # Build lookup for quick access
    clip_lookup = {str(m["id"]): m for m in clips}

    # Create ActivityCluster objects
    results: list[ActivityCluster] = []
    for clip_ids in raw_clusters:
        cluster_clips = [clip_lookup[cid] for cid in clip_ids]

        # Compute time range
        times = [_parse_iso(str(c["created_at"])) for c in cluster_clips]
        time_start = min(times).isoformat()
        time_end = max(times).isoformat()

        # Compute GPS center (mean of non-null coords)
        lats = [float(c["gps_lat"]) for c in cluster_clips if c.get("gps_lat") is not None]
        lons = [float(c["gps_lon"]) for c in cluster_clips if c.get("gps_lon") is not None]
        gps_center_lat = sum(lats) / len(lats) if lats else None
        gps_center_lon = sum(lons) / len(lons) if lons else None

        cluster_id = str(uuid.uuid4())
        ac = ActivityCluster(
            cluster_id=cluster_id,
            clip_ids=clip_ids,
            time_start=time_start,
            time_end=time_end,
            gps_center_lat=gps_center_lat,
            gps_center_lon=gps_center_lon,
        )
        results.append(ac)

        # Store in DB
        db.insert_activity_cluster(
            cluster_id,
            time_start=time_start,
            time_end=time_end,
            gps_center_lat=gps_center_lat,
            gps_center_lon=gps_center_lon,
            clip_ids_json=json.dumps(clip_ids),
        )

    logger.info("Created %d activity clusters", len(results))
    return results
