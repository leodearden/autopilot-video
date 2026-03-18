"""Interactive face cluster labeling script.

Provides helper functions for querying unlabeled face clusters and
applying human-readable labels, plus an interactive CLI loop.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB


def get_unlabeled_clusters(db: CatalogDB) -> list[dict]:
    """Return face clusters that have no label (label IS NULL).

    Args:
        db: Catalog database with face_clusters table.

    Returns:
        List of cluster dicts with label == None.
    """
    clusters = db.get_face_clusters()
    return [c for c in clusters if c["label"] is None]


def apply_label(db: CatalogDB, cluster_id: int, label: str) -> None:
    """Set a human-readable label on a face cluster.

    Args:
        db: Catalog database.
        cluster_id: The cluster to label.
        label: Human-readable name (e.g. "Alice").
    """
    with db:
        db.update_face_label(cluster_id, label)


def main(db_path: str) -> None:
    """Interactive CLI loop for labeling face clusters.

    Args:
        db_path: Path to the SQLite catalog database.
    """
    from autopilot.db import CatalogDB

    db = CatalogDB(db_path)
    try:
        unlabeled = get_unlabeled_clusters(db)
        if not unlabeled:
            print("All face clusters are already labeled.")
            return

        print(f"Found {len(unlabeled)} unlabeled cluster(s).\n")

        for cluster in unlabeled:
            cid = cluster["cluster_id"]
            paths = json.loads(cluster["sample_image_paths"] or "[]")
            print(f"Cluster {cid}:")
            print(f"  Sample frames: {paths}")

            label = input("  Enter name (or 'skip' to skip): ").strip()
            if label and label.lower() != "skip":
                apply_label(db, cid, label)
                print(f"  Labeled cluster {cid} as '{label}'")
            else:
                print(f"  Skipped cluster {cid}")
            print()

        print("Done labeling.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label face clusters interactively")
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the SQLite catalog database",
    )
    args = parser.parse_args()
    main(args.db_path)
