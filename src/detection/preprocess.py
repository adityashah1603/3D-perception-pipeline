import numpy as np
from pathlib import Path


def load_pointcloud(file_path: str) -> np.ndarray:
    """Load point cloud from .bin (nuScenes) or ASCII .pcd file."""
    path = Path(file_path)
    if path.suffix == ".bin":
        # nuScenes LiDAR files: (N, 5) — x, y, z, intensity, ring_index
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    elif path.suffix == ".pcd":
        return _load_pcd_ascii(file_path)
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")


def _load_pcd_ascii(file_path: str) -> np.ndarray:
    """Load ASCII PCD file, returning (N, D) float32 array."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("DATA"):
            if "binary" in line:
                raise NotImplementedError(
                    "Binary PCD not supported; convert to ASCII or use .bin files."
                )
            header_end = i + 1
            break

    data = np.loadtxt(lines[header_end:], dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def filter_pointcloud(
    points: np.ndarray,
    x_range: tuple = (-51.2, 51.2),
    y_range: tuple = (-51.2, 51.2),
    z_range: tuple = (-5.0, 3.0),
) -> np.ndarray:
    """Remove points outside the CenterPoint detection range."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def normalize_intensity(points: np.ndarray) -> np.ndarray:
    """Normalize intensity channel (index 3) to [0, 1]."""
    if points.shape[1] < 4:
        return points
    points = points.copy()
    max_val = points[:, 3].max()
    if max_val > 0:
        points[:, 3] /= max_val
    return points


def preprocess(
    file_path: str,
    x_range: tuple = (-51.2, 51.2),
    y_range: tuple = (-51.2, 51.2),
    z_range: tuple = (-5.0, 3.0),
) -> np.ndarray:
    """Full preprocessing: load → range filter → intensity normalize."""
    points = load_pointcloud(file_path)
    points = filter_pointcloud(points, x_range, y_range, z_range)
    points = normalize_intensity(points)
    return points
