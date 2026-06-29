"""Training entry point for CenterPoint on nuScenes."""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Train CenterPoint on nuScenes")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--work-dir", default="work_dirs/centerpoint")
    return parser.parse_args()


def build_mmdet3d_override(cfg: dict, work_dir: str) -> str:
    """
    Generate a minimal mmdet3d Python config that overrides the base config
    with dataset paths and hyperparameters from our YAML.
    """
    base = cfg["model"]["base_config"]
    data_root = cfg["data"]["root"]
    version = cfg["data"]["version"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    max_epochs = cfg["train"]["max_epochs"]
    save_interval = cfg["checkpoint"]["save_interval"]

    py_cfg = f"""\
_base_ = ['{base}']

data_root = '{data_root}/'

train_dataloader = dict(
    batch_size={batch_size},
    num_workers={num_workers},
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        version='{version}',
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers={num_workers},
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        version='{version}',
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'nuscenes_infos_val.pkl', version='{version}')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs={max_epochs}, val_interval=1)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval={save_interval}))
"""
    out = Path(work_dir) / "centerpoint_override.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(py_cfg)
    return str(out)


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
    print(f"[train] Generated mmdet3d config: {py_cfg_path}")

    runner_cfg = dict(
        cfg_file=py_cfg_path,
        work_dir=args.work_dir,
        resume=args.resume is not None,
        load_from=args.resume,
    )
    runner = Runner.from_cfg(runner_cfg)
    runner.train()

    # Copy final checkpoint to Google Drive when running on Colab
    drive_dir = cfg["checkpoint"].get("drive_dir", "")
    if drive_dir and Path(drive_dir).exists():
        final_ckpt = Path(args.work_dir) / f"epoch_{cfg['train']['max_epochs']}.pth"
        if final_ckpt.exists():
            shutil.copy(final_ckpt, drive_dir)
            print(f"[train] Checkpoint saved to Drive: {drive_dir}")


if __name__ == "__main__":
    main()
