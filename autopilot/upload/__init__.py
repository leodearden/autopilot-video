"""YouTube upload and thumbnail extraction for autopilot-video."""

from autopilot.upload.thumbnail import ThumbnailError, extract_best_thumbnail
from autopilot.upload.youtube import UploadError, upload_video

__all__ = [
    "ThumbnailError",
    "UploadError",
    "extract_best_thumbnail",
    "upload_video",
]
