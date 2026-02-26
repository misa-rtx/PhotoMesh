from __future__ import annotations
import numpy as np
import cv2

def inpaint_atlas(
    atlas: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea",
) -> np.ndarray:
    """Fill empty texels in the texture atlas using OpenCV inpainting.

    Parameters
    ----------
    atlas : np.ndarray
        ``(H, W, 3)`` uint8 texture atlas.
    mask : np.ndarray
        ``(H, W)`` bool — ``True`` where a texel has colour.
    radius : int
        Inpainting neighbourhood radius (pixels).
    method : str
        ``"telea"`` (default) or ``"ns"`` (Navier-Stokes).

    Returns
    -------
    np.ndarray
        Inpainted atlas (same shape/dtype).
    """

    empty_mask = (~mask).astype(np.uint8) * 255
    n_empty = int(empty_mask.sum()) // 255

    if n_empty == 0:
        return atlas

    methods = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown inpaint method '{method}'. Choose from: {list(methods)}"
        )

    atlas_bgr = cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR)
    atlas_bgr = cv2.inpaint(atlas_bgr, empty_mask, radius, methods[method])
    return cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
