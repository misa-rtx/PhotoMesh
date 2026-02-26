from __future__ import annotations
import numpy as np


def bilinear_sample(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Vectorised bilinear interpolation from *image* at sub-pixel ``(u, v)``.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` float32 image.
    u, v : np.ndarray
        ``(K,)`` horizontal / vertical pixel coordinates.

    Returns
    -------
    np.ndarray
        ``(K, 3)`` sampled colours (same dtype as *image*).
    """
    H, W = image.shape[:2]
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)

    xf = u - np.floor(u)
    yf = v - np.floor(v)

    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]

    sampled = (
        c00 * ((1 - xf) * (1 - yf))[:, None]
        + c10 * (xf * (1 - yf))[:, None]
        + c01 * ((1 - xf) * yf)[:, None]
        + c11 * (xf * yf)[:, None]
    )
    return sampled.astype(image.dtype)