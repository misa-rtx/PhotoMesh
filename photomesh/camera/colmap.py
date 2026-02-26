from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def parse_colmap_cameras(
    path: str,
) -> Dict[int, Tuple[str, int, int, np.ndarray]]:
    """Parse a COLMAP ``cameras.txt`` file.

    Returns
    -------
    dict
        ``{camera_id: (model, width, height, params)}`` where *params* is a
        1-D float array whose contents depend on the camera model.
    """
    cameras: Dict[int, Tuple[str, int, int, np.ndarray]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]])
            cameras[cam_id] = (model, width, height, params)
    return cameras


def parse_colmap_images(
    path: str,
) -> List[Tuple[int, np.ndarray, np.ndarray, int, str]]:
    """Parse a COLMAP ``images.txt`` file.

    Returns
    -------
    list
        Sorted (by image_id) list of
        ``(image_id, qvec, tvec, camera_id, name)`` tuples.
    """
    images: List[Tuple[int, np.ndarray, np.ndarray, int, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Skip POINTS2D lines (fewer than 10 tokens)
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]])
            tvec = np.array([float(x) for x in parts[5:8]])
            camera_id = int(parts[8])
            name = parts[9]
            images.append((image_id, qvec, tvec, camera_id, name))
    images.sort(key=lambda x: x[0])
    return images


def load_colmap_dataset(
    dataset_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a COLMAP dataset directory and return camera arrays.

    The directory must contain ``cameras.txt``, ``images.txt``, and an
    ``images/`` sub-folder with the referenced image files.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory.

    Returns
    -------
    cam_R : np.ndarray
        ``(N, 3, 3)`` world-to-camera rotation matrices.
    cam_t : np.ndarray
        ``(N, 3)`` world-to-camera translation vectors.
    cam_intr : np.ndarray
        ``(N, 6)`` intrinsics per view: ``[fx, fy, cx, cy, W, H]``.
    image_paths : list of str
        Absolute image file paths (order matches the camera arrays).
    """
    cameras = parse_colmap_cameras(os.path.join(dataset_dir, "cameras.txt"))
    images = parse_colmap_images(os.path.join(dataset_dir, "images.txt"))

    n = len(images)
    cam_R = np.zeros((n, 3, 3), dtype=np.float64)
    cam_t = np.zeros((n, 3), dtype=np.float64)
    cam_intr = np.zeros((n, 6), dtype=np.float64)
    image_paths: List[str] = []

    for i, (image_id, qvec, tvec, camera_id, name) in enumerate(images):
        model, width, height, params = cameras[camera_id]
        if model != "PINHOLE":
            raise ValueError(
                f"Only PINHOLE camera model is supported, got '{model}' "
                f"for camera {camera_id}"
            )
        fx, fy, cx, cy = params[:4]

        cam_R[i] = _qvec2rotmat(qvec)
        cam_t[i] = tvec
        cam_intr[i] = [fx, fy, cx, cy, width, height]
        image_paths.append(os.path.join(dataset_dir, name))

    return cam_R, cam_t, cam_intr, image_paths
