"""BEVFusion camera + LiDAR fusion — Stage 2 (placeholder)."""
from __future__ import annotations

from typing import Optional

try:
    from mmdet3d.apis import init_model, inference_detector
    MMDET3D_AVAILABLE = True
except ImportError:
    MMDET3D_AVAILABLE = False


class BEVFusionDetector:
    """Wrapper around mmdetection3d's BEVFusion model (camera + LiDAR in BEV)."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        if not MMDET3D_AVAILABLE:
            raise ImportError("mmdet3d is not installed.")
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.model.eval()

    def detect(self, sample: dict) -> dict:
        """
        Run fused detection on a sample dict containing:
          - pcd_path: str
          - img_paths: list[str]  (6 camera images for nuScenes)

        Returns dict with boxes_3d, scores_3d, labels_3d.
        """
        raise NotImplementedError("BEVFusion inference — Stage 2, coming soon.")
