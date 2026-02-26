from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class UVParametrizer(ABC):
    """Protocol for UV-unwrapping a triangle mesh.

    Subclasses must implement :meth:`parametrize`, which takes raw
    vertices and triangles and returns unwrapped UV coordinates plus
    new (possibly duplicated-at-seams) vertex/triangle arrays.
    """

    @abstractmethod
    def parametrize(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute UV parametrization.

        Parameters
        ----------
        vertices : np.ndarray
            ``(V, 3)`` vertex positions.
        triangles : np.ndarray
            ``(F, 3)`` triangle indices.

        Returns
        -------
        new_verts : np.ndarray
            ``(V', 3)`` vertices (possibly with seam duplicates).
        new_indices : np.ndarray
            ``(F', 3)`` triangle indices into *new_verts*.
        new_uvs : np.ndarray
            ``(V', 2)`` UV coordinates normalised to ``[0, 1]``.
        vmapping : np.ndarray
            ``(V',)`` mapping from new vertex indices to original indices.
        """
        ...
