"""On-demand video analysis for SiteGuardVision.

The image pipeline in :mod:`infer_engine` remains the single source of truth for
model loading, label mapping, geometry, and PPE association.  This package adds
video sampling, worker tracking, temporal event verification, annotated output,
and a safe sample-video library on top of it.
"""

from __future__ import annotations

__all__ = ["sample_library", "video_engine", "jobs"]
