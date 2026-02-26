"""xatlas-based UV parametrization."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from photomesh.uv.base import UVParametrizer


class XAtlasParametrizer(UVParametrizer):
    """UV unwrap using the `xatlas <https://github.com/mworchel/xatlas-python>`_ library.

    This is the default parametrizer used by photomesh.  It produces a
    packed UV atlas with seam-duplicated vertices ready for texture baking.
    """

    def parametrize(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run xatlas and return normalised UVs.

        Returns
        -------
        new_verts, new_indices, new_uvs, vmapping
        """
        import xatlas

        vmapping, new_indices, new_uvs = xatlas.parametrize(
            vertices.astype(np.float32),
            triangles.astype(np.uint32),
        )

        # Normalise UVs to [0, 1]
        uv_min = new_uvs.min(axis=0)
        uv_max = new_uvs.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range == 0] = 1.0
        new_uvs_norm = (new_uvs - uv_min) / uv_range

        new_verts = vertices[vmapping]

        return new_verts, new_indices, new_uvs_norm, vmapping
