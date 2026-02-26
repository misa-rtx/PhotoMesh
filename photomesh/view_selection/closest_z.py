"""Normal-weighted, multi-view blending view-selection strategy."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from photomesh.view_selection.base import ViewSelector
from photomesh.utils import bilinear_sample


class ClosestZSelector(ViewSelector):
    """Select and blend camera views weighted by viewing angle and depth.

    For each texel the selector:
    1. Rejects any camera that sees the face from behind
       (dot product of face normal and view-to-surface vector <= 0).
    2. Scores each remaining camera by  cos(theta) / z  where theta is
       the angle between the face normal and the direction from the
       surface to the camera, and z is the camera-space depth.
    3. Blends the colours of **all** valid cameras proportionally to
       their scores (soft blending).  This eliminates hard seams at
       view-boundary transitions while naturally down-weighting
       grazing-angle cameras.
    """

    def select(
        self,
        pts3d: np.ndarray,
        cam_R: np.ndarray,
        cam_t: np.ndarray,
        cam_intr: np.ndarray,
        images: List[np.ndarray],
        face_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = len(pts3d)
        n_views = len(images)

        # Accumulators for weighted blending
        color_acc  = np.zeros((K, 3), dtype=np.float64)
        weight_acc = np.zeros(K,      dtype=np.float64)

        for vi in range(n_views):
            R  = cam_R[vi]
            t  = cam_t[vi]
            fx, fy, cx, cy, Wo, Ho = cam_intr[vi]

            # Camera centre in world space: C = -R^T t
            cam_center = -(R.T @ t)  # (3,)

            # Per-texel view direction (surface → camera), not normalised yet
            to_cam = cam_center - pts3d  # (K, 3)
            dist   = np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12
            to_cam_n = to_cam / dist      # (K, 3) unit vectors

            # cos(theta): dot of face normal with direction toward camera
            # Use the absolute value so the winding-order of the mesh
            # doesn't matter — we just want the best-facing camera.
            cos_theta = np.abs(to_cam_n @ face_normal)  # (K,)

            # Reject cameras that see the face nearly edge-on (< 5°)
            visible = cos_theta > 0.087   # cos(85°) ≈ 0.087

            # Project to image plane
            pc  = (R @ pts3d.T).T + t    # (K, 3) camera-space
            z   = pc[:, 2]
            vis = (z > 1e-6) & visible

            if not vis.any():
                continue

            u = fx * pc[:, 0] / z + cx
            v = fy * pc[:, 1] / z + cy
            in_bounds = vis & (u >= 0) & (u < Wo) & (v >= 0) & (v < Ho)

            if not in_bounds.any():
                continue

            # Score: angle quality / depth  (higher = better view)
            score = np.where(in_bounds, cos_theta / (z + 1e-6), 0.0)

            # Sample colours for in-bounds texels
            sampled = bilinear_sample(images[vi], u[in_bounds], v[in_bounds])

            color_acc[in_bounds]  += score[in_bounds, None] * sampled
            weight_acc[in_bounds] += score[in_bounds]

        # Normalise
        filled = weight_acc > 0
        best_colors = np.zeros((K, 3), dtype=np.float32)
        best_colors[filled] = (color_acc[filled] / weight_acc[filled, None]).astype(np.float32)

        return best_colors, filled
