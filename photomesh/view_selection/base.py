"""Abstract base class for view-selection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class ViewSelector(ABC):
    """Protocol for choosing which camera view(s) contribute colour to each texel.

    Subclasses must implement :meth:`select`, which receives a batch of
    3-D surface points and returns the best colour for each point by
    evaluating all available camera views.
    """

    @abstractmethod
    def select(
        self,
        pts3d: np.ndarray,
        cam_R: np.ndarray,
        cam_t: np.ndarray,
        cam_intr: np.ndarray,
        images: List[np.ndarray],
        face_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Choose colours for a batch of 3-D points.

        Parameters
        ----------
        pts3d : np.ndarray
            ``(K, 3)`` world-space surface points to colour.
        cam_R : np.ndarray
            ``(N, 3, 3)`` rotation matrices (world-to-camera).
        cam_t : np.ndarray
            ``(N, 3)`` translation vectors (world-to-camera).
        cam_intr : np.ndarray
            ``(N, 6)`` intrinsics per view:
            ``[fx, fy, cx, cy, W_orig, H_orig]``.
        images : list of np.ndarray
            ``N`` images, each ``(H, W, 3)`` float32 in ``[0, 255]``.
        face_normal : np.ndarray
            ``(3,)`` unit normal of the current triangle in world space.

        Returns
        -------
        best_colors : np.ndarray
            ``(K, 3)`` float32 colours in ``[0, 255]``.
        best_mask : np.ndarray
            ``(K,)`` bool — ``True`` where a valid colour was found.
        """
        ...
