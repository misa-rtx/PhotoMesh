"""Closest-Z (nearest camera) view-selection strategy."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from photomesh.view_selection.base import ViewSelector
from photomesh.utils import bilinear_sample


class ClosestZSelector(ViewSelector):
    """Select the camera view with the smallest positive depth for each texel.

    This corresponds to picking the camera whose optical centre is
    closest to the surface point (along the viewing ray), which
    typically yields the sharpest texture.  It is the default strategy
    used by photomesh.
    """

    def select(
        self,
        pts3d: np.ndarray,
        cam_R: np.ndarray,
        cam_t: np.ndarray,
        cam_intr: np.ndarray,
        images: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = len(pts3d)
        n_views = len(images)

        best_colors = np.zeros((K, 3), dtype=np.float32)
        best_scores = np.full(K, -1.0, dtype=np.float64)

        for vi in range(n_views):
            R = cam_R[vi]
            t = cam_t[vi]
            fx, fy, cx, cy, Wo, Ho = cam_intr[vi]

            pc = (R @ pts3d.T).T + t
            z = pc[:, 2]
            vis = z > 1e-6
            if not vis.any():
                continue

            u = fx * pc[:, 0] / z + cx
            v = fy * pc[:, 1] / z + cy
            in_bounds = vis & (u >= 0) & (u < Wo) & (v >= 0) & (v < Ho)
            if not in_bounds.any():
                continue

            score = np.where(in_bounds, 1.0 / (z + 1e-6), -1.0)
            better = score > best_scores
            if not better.any():
                continue

            sampled = bilinear_sample(images[vi], u[better], v[better])
            best_colors[better] = sampled
            best_scores[better] = score[better]

        best_mask = best_scores > 0
        return best_colors, best_mask
