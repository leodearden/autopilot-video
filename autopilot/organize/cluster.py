"""Activity clustering using temporal-spatial DBSCAN and semantic refinement.

Provides cluster_activities() for grouping media files into activity clusters
based on timestamps, GPS coordinates, and SigLIP embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

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
    excluded: bool = False


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

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
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
    clips: list[dict[str, Any]],
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

    # Parse timestamps into array
    ts_arr = np.array(
        [_parse_iso(str(clip["created_at"])).timestamp() for clip in clips],
        dtype=np.float64,
    )

    # Temporal distance matrix (normalized by threshold)
    t_norm = np.abs(np.subtract.outer(ts_arr, ts_arr)) / temporal_threshold_s

    # Extract GPS coordinates; use NaN for missing values
    lats = np.array(
        [float(c["gps_lat"]) if c.get("gps_lat") is not None else np.nan for c in clips],
        dtype=np.float64,
    )
    lons = np.array(
        [float(c["gps_lon"]) if c.get("gps_lon") is not None else np.nan for c in clips],
        dtype=np.float64,
    )

    # Vectorized haversine distance matrix
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = np.subtract.outer(lat_r, lat_r)
    dlon = np.subtract.outer(lon_r, lon_r)
    a = (
        np.sin(dlat / 2) ** 2
        + np.multiply.outer(np.cos(lat_r), np.cos(lat_r)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    s_dist = _EARTH_RADIUS_M * c
    s_norm = s_dist / spatial_threshold_m

    # Mask: both clips must have valid GPS for spatial distance
    has_gps = ~np.isnan(lats)
    both_have_gps = np.outer(has_gps, has_gps)

    # Combined: max(temporal, spatial) where both have GPS, else temporal-only
    dist_matrix = np.where(both_have_gps, np.maximum(t_norm, s_norm), t_norm)

    # DBSCAN with eps=1.0 (since distances are normalized to threshold)
    db = DBSCAN(eps=1.0, min_samples=1, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    # Group clips by label
    cluster_map: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        clip_id = str(clips[idx]["id"])
        lbl = int(label)
        if lbl not in cluster_map:
            cluster_map[lbl] = []
        cluster_map[lbl].append(clip_id)

    return list(cluster_map.values())


def _semantic_refine(
    clip_ids: list[str],
    clips_data: Mapping[str, Mapping[str, Any]],
    db: CatalogDB,
    *,
    cosine_threshold: float = 0.5,
) -> list[list[str]]:
    """Refine a cluster using SigLIP embedding discontinuities.

    Loads embeddings from the DB for the given clip IDs, sorts by timestamp,
    computes cosine distance between consecutive clips' mean embeddings, and
    splits at points where the distance exceeds the threshold.

    Args:
        clip_ids: List of clip IDs in the cluster.
        clips_data: Mapping from clip ID to media dict (needs 'created_at').
        db: Catalog database for loading embeddings.
        cosine_threshold: Cosine distance threshold for split detection.

    Returns:
        List of sub-clusters (each a list of clip IDs).
    """
    if len(clip_ids) <= 1:
        return [clip_ids]

    import numpy as np  # type: ignore[reportMissingImports]

    # Sort clips by timestamp
    sorted_ids = sorted(
        clip_ids,
        key=lambda cid: _parse_iso(str(clips_data[cid]["created_at"])).timestamp(),
    )

    # Load mean embedding per clip
    clip_embeddings: list[tuple[str, Any]] = []
    for cid in sorted_ids:
        rows = db.get_embeddings_for_media(cid)
        if not rows:
            clip_embeddings.append((cid, None))
            continue
        embs = [np.frombuffer(cast(bytes, row["embedding"]), dtype=np.float32) for row in rows]
        mean_emb = np.mean(embs, axis=0)
        clip_embeddings.append((cid, mean_emb))

    # Find split points based on cosine distance discontinuities
    split_points: list[int] = []
    for i in range(len(clip_embeddings) - 1):
        emb_a = clip_embeddings[i][1]
        emb_b = clip_embeddings[i + 1][1]
        if emb_a is None or emb_b is None:
            continue
        # Cosine distance = 1 - cosine similarity
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            continue
        cos_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        cos_dist = 1.0 - cos_sim
        if cos_dist > cosine_threshold:
            split_points.append(i + 1)

    if not split_points:
        return [sorted_ids]

    # Split at detected boundaries
    sub_clusters: list[list[str]] = []
    prev = 0
    for sp in split_points:
        sub_clusters.append(sorted_ids[prev:sp])
        prev = sp
    sub_clusters.append(sorted_ids[prev:])

    return sub_clusters


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

    # Phase 1: Temporal-spatial clustering
    raw_clusters = _temporal_spatial_cluster(clips)

    # Build lookup for quick access
    clip_lookup: dict[str, dict[str, Any]] = {str(m["id"]): m for m in clips}

    # Phase 2: Semantic refinement
    refined_clusters: list[list[str]] = []
    for cluster_ids in raw_clusters:
        sub_clusters = _semantic_refine(cluster_ids, clip_lookup, db)
        refined_clusters.extend(sub_clusters)

    # Create ActivityCluster objects (compute all before writing)
    results: list[ActivityCluster] = []
    for clip_ids in refined_clusters:
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

    # Atomic clear+insert: only wipe old data after computation succeeds
    with db.conn:
        db.clear_activity_clusters()
        for ac in results:
            db.insert_activity_cluster(
                ac.cluster_id,
                time_start=ac.time_start,
                time_end=ac.time_end,
                gps_center_lat=ac.gps_center_lat,
                gps_center_lon=ac.gps_center_lon,
                clip_ids_json=json.dumps(ac.clip_ids),
            )

    logger.info("Created %d activity clusters", len(results))
    return results
