from __future__ import annotations
from typing import List
import numpy as np
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer



# Available method names exposed for validation
AVAILABLE_METHODS = ("mkl", "hm-mkl-hm", "hm", "reinhard", "default")


def match_colors(
    images: List[np.ndarray],
    reference_index: int = 0,
    method: str = "mkl",
) -> List[np.ndarray]:
    """Match the colour distribution of every image to a reference.

    Parameters
    ----------
    images : list[np.ndarray]
        List of ``(H, W, 3)`` float32 images with values in ``[0, 255]``.
    reference_index : int
        Index of the reference image (default ``0`` = first image).
    method : str
        Transfer algorithm — one of ``"mkl"``, ``"hm-mkl-hm"``,
        ``"hm"``, ``"reinhard"``, ``"default"`` (default ``"mkl"``).

    Returns
    -------
    list[np.ndarray]
        Same-shaped images with matched colours (float32, ``[0, 255]``).
    """
   
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown color_match method '{method}'. "
            f"Available: {list(AVAILABLE_METHODS)}"
        )

    ref_img = np.clip(images[reference_index], 0, 255).astype(np.uint8)
    cm = ColorMatcher()

    out: List[np.ndarray] = []
    for i, img in enumerate(images):
        if i == reference_index:
            out.append(img.copy())
            continue

        src_img = np.clip(img, 0, 255).astype(np.uint8)
        matched = cm.transfer(src=src_img, ref=ref_img, method=method)
        matched = Normalizer(matched).uint8_norm()
        out.append(matched.astype(np.float32))

    return out
