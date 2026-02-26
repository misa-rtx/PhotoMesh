from __future__ import annotations
from typing import List, Tuple
import numpy as np
from photomesh.view_selection.base import ViewSelector


def rasterize(
    new_verts: np.ndarray,
    new_indices: np.ndarray,
    new_uvs_norm: np.ndarray,
    texture_size: int,
    view_selector: ViewSelector,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    cam_intr: np.ndarray,
    images: List[np.ndarray],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise every triangle in UV space and back-project colours.

    For each triangle, compute the bounding box in the UV atlas,
    iterate over texels inside the triangle (via barycentric coordinates),
    interpolate the 3-D position, and delegate colour selection to the
    :class:`~photomesh.view_selection.base.ViewSelector`.

    Parameters
    ----------
    new_verts : np.ndarray       ``(V', 3)``
    new_indices : np.ndarray     ``(F', 3)``
    new_uvs_norm : np.ndarray    ``(V', 2)``  in ``[0, 1]``
    texture_size : int
    view_selector : ViewSelector
    cam_R : np.ndarray           ``(N, 3, 3)``
    cam_t : np.ndarray           ``(N, 3)``
    cam_intr : np.ndarray        ``(N, 6)``
    images : list[np.ndarray]
    verbose : bool

    Returns
    -------
    atlas : np.ndarray  ``(texture_size, texture_size, 3)`` uint8
    mask  : np.ndarray  ``(texture_size, texture_size)`` bool
    """
    atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    mask = np.zeros((texture_size, texture_size), dtype=bool)

    # UV pixel coords for all vertices
    uv_px = new_uvs_norm * (texture_size - 1)  # (V', 2)

    filled_total = 0
    n_tris = len(new_indices)

    for fi in range(n_tris):
        if verbose and fi % 4000 == 0:
            pct = 100.0 * fi / n_tris if n_tris else 0
            print(
                f"    tri {fi}/{n_tris} ({pct:.0f}%) — "
                f"{filled_total} texels filled"
            )

        i0, i1, i2 = new_indices[fi]

        # Triangle corners in UV-pixel space and 3-D
        uv_a, uv_b, uv_c = uv_px[i0], uv_px[i1], uv_px[i2]
        p3d_a, p3d_b, p3d_c = new_verts[i0], new_verts[i1], new_verts[i2]

        # Bounding box
        umin = max(int(np.floor(min(uv_a[0], uv_b[0], uv_c[0]))), 0)
        umax = min(int(np.ceil(max(uv_a[0], uv_b[0], uv_c[0]))), texture_size - 1)
        vmin = max(int(np.floor(min(uv_a[1], uv_b[1], uv_c[1]))), 0)
        vmax = min(int(np.ceil(max(uv_a[1], uv_b[1], uv_c[1]))), texture_size - 1)

        if umin > umax or vmin > vmax:
            continue

        # Grid of candidate pixels
        xs = np.arange(umin, umax + 1, dtype=np.float64)
        ys = np.arange(vmin, vmax + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2)

        # Barycentric coordinates
        a = uv_a.astype(np.float64)
        b = uv_b.astype(np.float64)
        c = uv_c.astype(np.float64)
        v0 = c - a
        v1 = b - a
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue

        v2 = pts - a  # (M, 2)
        d02 = v2 @ v0  # (M,)
        d12 = v2 @ v1  # (M,)
        inv_d = 1.0 / denom
        bary_u = (d11 * d02 - d01 * d12) * inv_d
        bary_v = (d00 * d12 - d01 * d02) * inv_d
        bary_w = 1.0 - bary_u - bary_v

        # Inside-triangle mask (with small tolerance)
        inside = (bary_w >= -1e-4) & (bary_u >= -1e-4) & (bary_v >= -1e-4)
        if not inside.any():
            continue

        bary_w = bary_w[inside]
        bary_v = bary_v[inside]
        bary_u = bary_u[inside]
        px_coords = pts[inside].astype(np.int32)  # (K, 2)

        # Interpolate 3-D positions
        pts3d = (
            bary_w[:, None] * p3d_a
            + bary_v[:, None] * p3d_b
            + bary_u[:, None] * p3d_c
        )

        # Delegate colour selection to the ViewSelector
        best_colors, got_color = view_selector.select(
            pts3d, cam_R, cam_t, cam_intr, images,
        )

        # Write to atlas
        if got_color.any():
            px_good = px_coords[got_color]
            col_good = np.clip(best_colors[got_color], 0, 255).astype(np.uint8)
            atlas[px_good[:, 1], px_good[:, 0]] = col_good
            mask[px_good[:, 1], px_good[:, 0]] = True
            filled_total += int(got_color.sum())

    if verbose:
        print(f"  Filled {filled_total} texels total")

    return atlas, mask
