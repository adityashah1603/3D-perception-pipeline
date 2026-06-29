"""ByteTrack multi-object tracker — Stage 3 (placeholder)."""
from __future__ import annotations

import numpy as np


class ByteTracker:
    """
    Wraps ByteTrack to associate 3D detection boxes across frames.
    Operates on BEV (bird's-eye-view) x/y centroids + scores.
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self._tracks: list[dict] = []
        self._next_id = 0

    def update(self, detections: dict, frame_id: int) -> list[dict]:
        """
        Associate new detections with existing tracks.

        Args:
            detections: output of CenterPointDetector.detect()
            frame_id:   current frame index

        Returns:
            list of active tracks, each dict has:
              track_id, box_3d, score, label, age
        """
        raise NotImplementedError("ByteTrack integration — Stage 3, coming soon.")
