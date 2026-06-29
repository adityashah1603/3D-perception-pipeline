"""Evaluate a CenterPoint checkpoint on the nuScenes val split."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train import build_mmdet3d_override


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CenterPoint on nuScenes")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--work-dir", default="work_dirs/eval")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        from mmdet3d.utils import register_all_modules
        from mmengine.runner import Runner
        register_all_modules()
    except ImportError as e:
        sys.exit(f"mmdet3d not found — install dependencies first.\n{e}")

    py_cfg_path = build_mmdet3d_override(cfg, args.work_dir)
    print(f"[eval] Config: {py_cfg_path}")
    print(f"[eval] Checkpoint: {args.checkpoint}")

    runner = Runner.from_cfg(
        dict(
            cfg_file=py_cfg_path,
            work_dir=args.work_dir,
            load_from=args.checkpoint,
        )
    )
    metrics = runner.test()

    map_val = metrics.get("NuScenes metric/mAP", metrics.get("mAP", "N/A"))
    print(f"\n{'='*50}")
    print(f"  NuScenes mAP : {map_val}")
    print(f"{'='*50}")
    for k, v in sorted(metrics.items()):
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<45} {val}")


if __name__ == "__main__":
    main()
