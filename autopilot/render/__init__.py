"""Rendering pipeline — FFmpeg fast path, MoviePy complex path, and routing."""

from autopilot.render.crop import CropError, compute_crop_path
from autopilot.render.ffmpeg_render import RenderError, render_simple
from autopilot.render.moviepy_render import ComplexRenderError, render_complex
from autopilot.render.router import RoutingError, route_and_render
from autopilot.render.validate import ValidationError, validate_render

__all__ = [
    "ComplexRenderError",
    "CropError",
    "RenderError",
    "RoutingError",
    "ValidationError",
    "compute_crop_path",
    "render_complex",
    "render_simple",
    "route_and_render",
    "validate_render",
]
