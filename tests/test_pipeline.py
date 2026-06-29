"""Smoke tests for the perception pipeline components."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.preprocess import filter_pointcloud, normalize_intensity, preprocess


class TestPreprocess:
    def _make_points(self):
        rng = np.random.default_rng(42)
        return rng.uniform(-100, 100, (500, 5)).astype(np.float32)

    def test_filter_reduces_points(self):
        pts = self._make_points()
        filtered = filter_pointcloud(pts)
        assert len(filtered) < len(pts)

    def test_filter_range_respected(self):
        pts = self._make_points()
        filtered = filter_pointcloud(pts, x_range=(-10, 10), y_range=(-10, 10), z_range=(-2, 2))
        assert (filtered[:, 0] >= -10).all() and (filtered[:, 0] <= 10).all()

    def test_normalize_intensity_bounds(self):
        pts = self._make_points()
        pts[:, 3] = np.abs(pts[:, 3])  # ensure positive intensities
        normalized = normalize_intensity(pts)
        assert normalized[:, 3].max() <= 1.0 + 1e-6

    def test_normalize_zero_intensity_safe(self):
        pts = self._make_points()
        pts[:, 3] = 0.0
        result = normalize_intensity(pts)
        assert (result[:, 3] == 0.0).all()

    def test_preprocess_returns_ndarray(self, tmp_path):
        # Write a minimal .bin file
        pts = np.random.randn(100, 5).astype(np.float32)
        pcd_file = tmp_path / "test.bin"
        pts.tofile(pcd_file)
        result = preprocess(str(pcd_file))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2 and result.shape[1] == 5
