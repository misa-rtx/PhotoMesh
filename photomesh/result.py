from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np
import open3d as o3d
from photomesh import mesh_io


@dataclass
class TextureResult:
    """Container for all texture-mapping outputs.

    Holds the UV-unwrapped mesh geometry, the baked texture atlas,
    and a ready-to-use Open3D mesh.  Call :meth:`save` to export
    files, or access the arrays directly for further processing.

    Attributes
    ----------
    vertices : np.ndarray  ``(V, 3)``
        Vertex positions (seam-duplicated by the UV unwrap).
    triangles : np.ndarray ``(F, 3)``
        Triangle indices into *vertices*.
    uvs : np.ndarray       ``(V, 2)``
        Normalised UV coordinates in ``[0, 1]``.
    atlas : np.ndarray     ``(tex_size, tex_size, 3)``
        Baked uint8 texture atlas (RGB).
    mask : np.ndarray      ``(tex_size, tex_size)``
        Bool mask — ``True`` where a texel received colour.
    texture_size : int
        Resolution of the atlas (e.g. 4096).
    mesh : o3d.geometry.TriangleMesh
        Open3D mesh with vertex colours sampled from the atlas.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray
    atlas: np.ndarray
    mask: np.ndarray
    texture_size: int
    mesh: o3d.geometry.TriangleMesh = field(repr=False)

    def get_open3d_mesh(self) -> o3d.geometry.TriangleMesh:
        """Return the Open3D mesh (with vertex colours)."""
        return self.mesh

    def save(
        self,
        output_dir: str = "outputs",
        prefix: str = "textured",
        formats: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Export result files to *output_dir*.

        Parameters
        ----------
        output_dir : str
            Directory to write into (created if needed).
        prefix : str
            Filename prefix.
        formats : sequence of str, optional
            Which formats to write.  Defaults to ``["obj", "ply"]``.
            Supported: ``"obj"`` (OBJ + MTL + PNG), ``"ply"``.

        Returns
        -------
        list of str
            Paths of all written files.
        """
        if formats is None:
            formats = ["obj", "ply"]

        os.makedirs(output_dir, exist_ok=True)
        written: List[str] = []

        if "obj" in formats:
            obj, mtl, tex = mesh_io.save_obj(
                output_dir,
                prefix,
                self.vertices,
                self.triangles,
                self.uvs,
                self.atlas,
            )
            written.extend([obj, mtl, tex])

        if "ply" in formats:
            ply = mesh_io.save_ply_textured(
                output_dir,
                prefix,
                self.vertices,
                self.triangles,
                self.atlas,
                self.uvs,
            )
            written.append(ply)

        return written
