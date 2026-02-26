"""photomesh — texture mapping between camera frames and 3D meshes.

Usage::

    import photomesh

    result = photomesh.map_texture(
        dataset_dir="dataset/",
        mesh_name="poisson.ply",
    )
    result.save("outputs/")
"""

__version__ = "0.1.0"

from photomesh.color_match import match_colors
from photomesh.core import map_texture
from photomesh.result import TextureResult

# Abstract bases (for custom implementations)
from photomesh.uv.base import UVParametrizer
from photomesh.view_selection.base import ViewSelector

# Built-in implementations
from photomesh.camera.colmap import load_colmap_dataset
from photomesh.uv.xatlas_param import XAtlasParametrizer
from photomesh.view_selection.closest_z import ClosestZSelector

__all__ = [
    # Main API
    "map_texture",
    "match_colors",
    "TextureResult",
    # Abstract bases
    "UVParametrizer",
    "ViewSelector",
    # Built-in implementations
    "load_colmap_dataset",
    "XAtlasParametrizer",
    "ClosestZSelector",
]
