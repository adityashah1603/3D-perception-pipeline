from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from mmdet3d.apis import init_model, inference_detector
    MMDET3D_AVAILABLE = True
except ImportError:
    MMDET3D_AVAILABLE = False


class CenterPointDetector:
    """Thin wrapper around mmdetection3d's CenterPoint for inference."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        if not MMDET3D_AVAILABLE:
            raise ImportError(
                "mmdet3d is not installed. Run: pip install mmdet3d"
            )
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.model.eval()

    def detect(self, pcd_path: str) -> dict:
        """
        Run detection on a single point cloud file.

        Returns:
            dict with keys:
              - boxes_3d:  (N, 9) array — x,y,z,l,w,h,rot,vx,vy
              - scores_3d: (N,)   confidence scores
              - labels_3d: (N,)   class indices
        """
        result, _ = inference_detector(self.model, pcd_path)
        pred = result.pred_instances_3d
        return {
            "boxes_3d": pred.bboxes_3d.tensor.cpu().numpy(),
            "scores_3d": pred.scores_3d.cpu().numpy(),
            "labels_3d": pred.labels_3d.cpu().numpy(),
        }

    def detect_batch(self, pcd_paths: list[str]) -> list[dict]:
        return [self.detect(p) for p in pcd_paths]
