from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import open3d as o3d
from PIL import Image


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh and remove NaN/Inf vertices.

    Parameters
    ----------
    path : str
        Path to a mesh file readable by Open3D (e.g. ``.ply``).

    Returns
    -------
    vertices : np.ndarray  ``(V, 3)``
    triangles : np.ndarray ``(F, 3)``
    """
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).copy()

    valid = np.isfinite(vertices).all(axis=1)
    if not valid.all():
        old_to_new = np.full(len(vertices), -1, dtype=np.int32)
        old_to_new[valid] = np.arange(valid.sum())
        vertices = vertices[valid]
        keep = [
            old_to_new[t]
            for t in triangles
            if (old_to_new[t] >= 0).all()
        ]
        triangles = np.array(keep, dtype=np.int32) if keep else np.empty((0, 3), dtype=np.int32)

    return vertices, triangles


def save_obj(
    output_dir: str,
    prefix: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    atlas: np.ndarray,
) -> Tuple[str, str, str]:
    """Write textured mesh as OBJ + MTL + texture PNG.

    Parameters
    ----------
    output_dir : str
        Directory to write into (created if needed).
    prefix : str
        File-name prefix (e.g. ``"textured"``).
    vertices : np.ndarray  ``(V, 3)``
    triangles : np.ndarray ``(F, 3)``
    uvs : np.ndarray       ``(V, 2)``  normalised to ``[0, 1]``
    atlas : np.ndarray      ``(H, W, 3)`` uint8 texture image.

    Returns
    -------
    obj_path, mtl_path, tex_path : str
    """
    os.makedirs(output_dir, exist_ok=True)

    tex_file = f"{prefix}_texture.png"
    obj_file = f"{prefix}.obj"
    mtl_file = f"{prefix}.mtl"

    tex_path = os.path.join(output_dir, tex_file)
    obj_path = os.path.join(output_dir, obj_file)
    mtl_path = os.path.join(output_dir, mtl_file)

    # Texture image
    Image.fromarray(atlas).save(tex_path)

    # MTL
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_file}\n")

    # OBJ (V=0 is bottom in OBJ → flip V)
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_file}\n")
        f.write("usemtl material_0\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {1.0 - uv[1]}\n")
        for tri in triangles:
            i0, i1, i2 = tri + 1  # OBJ is 1-indexed
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

    return obj_path, mtl_path, tex_path


def save_ply_textured(
    output_dir: str,
    prefix: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    atlas: np.ndarray,
    uvs: np.ndarray,
) -> str:
    """Write a textured PLY with per-face UV coords and a texture PNG.

    This produces a PLY file that references an external texture image
    via ``comment TextureFile`` and stores per-face texture coordinates,
    giving full atlas-resolution colour in viewers like MeshLab.

    Returns
    -------
    ply_path : str
    """
    os.makedirs(output_dir, exist_ok=True)

    tex_file = f"{prefix}_texture.png"
    ply_path = os.path.join(output_dir, f"{prefix}.ply")
    tex_path = os.path.join(output_dir, tex_file)

    # Save texture image
    Image.fromarray(atlas).save(tex_path)

    n_verts = len(vertices)
    n_faces = len(triangles)

    with open(ply_path, "w", newline="\n") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment TextureFile {tex_file}\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property list uchar float texcoord\n")
        f.write("end_header\n")

        # Vertex positions
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Faces with per-face texture coordinates (V=0 at bottom, like OBJ)
        for tri in triangles:
            i0, i1, i2 = tri
            u0, v0 = uvs[i0][0], 1.0 - uvs[i0][1]
            u1, v1 = uvs[i1][0], 1.0 - uvs[i1][1]
            u2, v2 = uvs[i2][0], 1.0 - uvs[i2][1]
            f.write(
                f"3 {i0} {i1} {i2} "
                f"6 {u0} {v0} {u1} {v1} {u2} {v2}\n"
            )

    return ply_path



def build_open3d_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    atlas: np.ndarray,
    uvs: np.ndarray,
    texture_size: int,
) -> o3d.geometry.TriangleMesh:
    """Create an Open3D TriangleMesh with vertex colours from the atlas."""
    uv_pixel = uvs.copy()
    uv_pixel[:, 0] *= texture_size - 1
    uv_pixel[:, 1] *= texture_size - 1
    ux = np.clip(uv_pixel[:, 0].astype(np.int32), 0, texture_size - 1)
    uy = np.clip(uv_pixel[:, 1].astype(np.int32), 0, texture_size - 1)
    vert_colors = atlas[uy, ux].astype(np.float64) / 255.0

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    mesh.compute_vertex_normals()
    return mesh