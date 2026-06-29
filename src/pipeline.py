"""End-to-end pipeline: detection → fusion → tracking."""
import argparse
import sys
from pathlib import Path

import yaml

# Ensure repo root is on path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.centerpoint import CenterPointDetector
from src.detection.preprocess import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run full 3D perception pipeline")
    parser.add_argument("--input", required=True, help="Path to .bin or .pcd file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="CenterPoint checkpoint")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_detection(cfg: dict, pcd_path: str, checkpoint: str, device: str) -> dict:
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    detector = CenterPointDetector(
        config_path=cfg["model"]["base_config"],
        checkpoint_path=checkpoint,
        device=device,
    )
    return detector.detect(pcd_path)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[pipeline] Preprocessing: {args.input}")
    points = preprocess(args.input)
    print(f"[pipeline] {len(points)} points after filtering")

    print("[pipeline] Running CenterPoint detection ...")
    detections = run_detection(cfg, args.input, args.checkpoint, args.device)

    n = len(detections["scores_3d"])
    print(f"[pipeline] Detected {n} objects")
    for i in range(min(n, 5)):
        print(
            f"  [{i}] label={detections['labels_3d'][i]}  "
            f"score={detections['scores_3d'][i]:.3f}  "
            f"box={detections['boxes_3d'][i, :3]}"
        )

    # Stage 2 & 3: BEVFusion + ByteTrack — coming soon
    print("[pipeline] Done. (Fusion + tracking not yet enabled.)")


if __name__ == "__main__":
    main()
