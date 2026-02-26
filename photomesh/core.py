from __future__ import annotations
import os
from typing import Optional, Union
import numpy as np
from PIL import Image
from photomesh.camera.colmap import load_colmap_dataset
from photomesh.color_match import match_colors
from photomesh.inpaint import inpaint_atlas
from photomesh.mesh_io import build_open3d_mesh, load_mesh
from photomesh.rasterizer import rasterize
from photomesh.result import TextureResult
from photomesh.uv.base import UVParametrizer
from photomesh.uv.xatlas_param import XAtlasParametrizer
from photomesh.view_selection.base import ViewSelector
from photomesh.view_selection.closest_z import ClosestZSelector

_UV_METHODS = {
    "xatlas": XAtlasParametrizer,
}

_VIEW_SELECTORS = {
    "closest_z": ClosestZSelector,
}


def _subdivide_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    atlas: np.ndarray,
    texture_size: int,
    iterations: int,
    verbose: bool,
) -> tuple:
    """Midpoint subdivision that properly interpolates both positions and UVs."""
    verts = vertices.astype(np.float64)
    tris = triangles.astype(np.int32)
    uv = uvs.astype(np.float64)

    if verbose:
        print(f"  Subdividing: {len(verts)} verts, {len(tris)} tris")

    for it in range(iterations):
        edge_to_mid: dict = {}
        new_verts_list = list(verts)
        new_uvs_list = list(uv)
        new_tris_list = []

        for fi in range(len(tris)):
            i0, i1, i2 = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])

            mids = []
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                edge = (min(a, b), max(a, b))
                if edge not in edge_to_mid:
                    mid_idx = len(new_verts_list)
                    edge_to_mid[edge] = mid_idx
                    new_verts_list.append((verts[a] + verts[b]) * 0.5)
                    new_uvs_list.append((uv[a] + uv[b]) * 0.5)
                mids.append(edge_to_mid[edge])

            m01, m12, m20 = mids
            new_tris_list.append([i0, m01, m20])
            new_tris_list.append([m01, i1, m12])
            new_tris_list.append([m20, m12, i2])
            new_tris_list.append([m01, m12, m20])

        verts = np.array(new_verts_list, dtype=np.float64)
        uv = np.array(new_uvs_list, dtype=np.float64)
        tris = np.array(new_tris_list, dtype=np.int32)

        if verbose:
            print(f"  Iteration {it + 1}: {len(verts)} verts, {len(tris)} tris")

    if verbose:
        print(f"  After subdivision: {len(verts)} verts, {len(tris)} tris")

    return verts, tris, uv


def map_texture(
    dataset_dir: str,
    mesh_name: str = "poisson.ply",
    texture_size: int = 4096,
    uv_method: Union[UVParametrizer, str] = "xatlas",
    view_selector: Union[ViewSelector, str] = "closest_z",
    inpaint: bool = True,
    inpaint_radius: int = 3,
    color_match: bool = False,
    color_match_method: str = "mkl",
    color_match_ref: int = 0,
    subdivide: int = 0,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> TextureResult:
    """Map texture from camera frames onto a bare mesh.

    This is the main entry point of the **photomesh** library.  It
    orchestrates the full pipeline: mesh loading, COLMAP camera loading,
    UV unwrapping, per-texel back-projection, optional inpainting,
    and optional subdivision.

    Parameters
    ----------
    dataset_dir : str
        Path to a COLMAP dataset directory containing ``cameras.txt``,
        ``images.txt``, an ``images/`` sub-folder, and the mesh file.
    mesh_name : str
        Filename of the ``.ply`` mesh inside *dataset_dir* (default
        ``"poisson.ply"``).
    texture_size : int
        Resolution of the square texture atlas (default 4096).
    uv_method : str or UVParametrizer
        UV parametrization method.  ``"xatlas"`` (default) or a custom
        :class:`~photomesh.uv.base.UVParametrizer`.
    view_selector : str or ViewSelector
        View-selection strategy.  ``"closest_z"`` (default) or a custom
        :class:`~photomesh.view_selection.base.ViewSelector`.
    inpaint : bool
        Whether to inpaint empty texels (default ``True``).
    inpaint_radius : int
        Inpainting neighbourhood radius in pixels (default 3).
    color_match : bool
        Whether to match colour / illumination of all images to a
        reference frame before texture projection (default ``False``).
    color_match_method : str
        Algorithm for ``color-matcher``:  ``"mkl"`` (default),
        ``"hm-mkl-hm"``, ``"hm"``, ``"reinhard"``, ``"default"``.
    color_match_ref : int
        Index of the reference image for colour matching (default ``0``).
    subdivide : int
        Number of midpoint subdivision iterations (default 0 = none).
    output_dir : str, optional
        If given, automatically save results to this directory.
    verbose : bool
        Print progress messages (default ``True``).

    Returns
    -------
    TextureResult
        Dataclass with mesh, atlas, UVs, and a ``.save()`` method.

    Examples
    --------
    >>> import photomesh
    >>> result = photomesh.map_texture(
    ...     dataset_dir="dataset/",
    ...     mesh_name="poisson.ply",
    ... )
    >>> result.save("outputs/")
    """

    mesh_path = os.path.join(dataset_dir, mesh_name)

    if verbose:
        print("[1/4] Loading mesh ...")
    vertices, triangles = load_mesh(mesh_path)
    if verbose:
        print(f"  Mesh: {len(vertices)} verts, {len(triangles)} tris")

    if verbose:
        print("[2/4] Loading COLMAP cameras ...")
    cam_R, cam_t, cam_intr, image_paths = load_colmap_dataset(dataset_dir)
    n_views = len(image_paths)
    if verbose:
        print(f"  {n_views} cameras loaded")

    if isinstance(uv_method, str):
        if uv_method not in _UV_METHODS:
            raise ValueError(
                f"Unknown uv_method '{uv_method}'. "
                f"Available: {list(_UV_METHODS)}"
            )
        uv_param = _UV_METHODS[uv_method]()
    else:
        uv_param = uv_method

    if verbose:
        print("[3/4] UV-unwrapping ...")
    new_verts, new_indices, new_uvs, vmapping = uv_param.parametrize(
        vertices, triangles
    )
    if verbose:
        print(f"  UV: {len(new_verts)} verts, {len(new_indices)} tris")

    if verbose:
        print("[4/4] Loading images & projecting ...")

    loaded_images: list = []
    for i, path in enumerate(image_paths):
        img = np.array(
            Image.open(path).convert("RGB"), dtype=np.float32
        )
        loaded_images.append(img)

        H_img, W_img = img.shape[:2]
        W_cam, H_cam = cam_intr[i, 4], cam_intr[i, 5]
        if W_img != W_cam or H_img != H_cam:
            sx = W_img / W_cam
            sy = H_img / H_cam
            cam_intr[i, 0] *= sx   # fx
            cam_intr[i, 1] *= sy   # fy
            cam_intr[i, 2] *= sx   # cx
            cam_intr[i, 3] *= sy   # cy
            cam_intr[i, 4] = W_img
            cam_intr[i, 5] = H_img

    if verbose:
        print(f"  Loaded {n_views} images")

    if color_match:
        if verbose:
            print(f"  Matching colours to image {color_match_ref} "
                  f"(method={color_match_method}) ...")
        loaded_images = match_colors(
            loaded_images,
            reference_index=color_match_ref,
            method=color_match_method,
        )
        if verbose:
            print(f"  Colour matching complete")

    if isinstance(view_selector, str):
        if view_selector not in _VIEW_SELECTORS:
            raise ValueError(
                f"Unknown view_selector '{view_selector}'. "
                f"Available: {list(_VIEW_SELECTORS)}"
            )
        selector = _VIEW_SELECTORS[view_selector]()
    else:
        selector = view_selector

    if verbose:
        print(f"  Rasterising UV atlas ({texture_size}x{texture_size}) ...")

    atlas, mask = rasterize(
        new_verts,
        new_indices,
        new_uvs,
        texture_size,
        selector,
        cam_R,
        cam_t,
        cam_intr,
        loaded_images,
        verbose=verbose,
    )

    if inpaint:
        n_empty = int((~mask).sum())
        if verbose:
            print(f"  Inpainting {n_empty} empty texels ...")
        atlas = inpaint_atlas(atlas, mask, radius=inpaint_radius)

    final_verts = new_verts
    final_tris = new_indices
    final_uvs = new_uvs

    if subdivide > 0:
        if verbose:
            print(f"\n  Subdividing ({subdivide} iterations) ...")
        final_verts, final_tris, final_uvs = _subdivide_mesh(
            new_verts, new_indices, new_uvs, atlas, texture_size,
            subdivide, verbose,
        )
    elif verbose:
        print("\n  Building result ...")

    o3d_mesh = build_open3d_mesh(
        final_verts, final_tris, atlas, final_uvs, texture_size
    )

    result = TextureResult(
        vertices=final_verts,
        triangles=final_tris,
        uvs=final_uvs,
        atlas=atlas,
        mask=mask,
        texture_size=texture_size,
        mesh=o3d_mesh,
    )


    if output_dir is not None:
        written = result.save(output_dir)
        if verbose:
            for p in written:
                print(f"  Saved {p}")

    if verbose:
        print(f"\nDone — per-texel back-projection at {texture_size}x{texture_size}")

    return result
