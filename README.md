# PhotoMesh

Texture mapping between camera frames and 3D meshes using COLMAP camera poses.

**PhotoMesh** takes a bare mesh (`.ply`), a COLMAP dataset directory, and the corresponding images, then performs per-texel back-projection to produce a fully textured mesh (OBJ + texture atlas, PLY with vertex colours).

## How It Works

The pipeline runs in four stages:

1. **Mesh loading** — reads the `.ply` geometry (vertices + triangles).
2. **Camera loading** — parses COLMAP `cameras.txt` / `images.txt` to get pinhole intrinsics and world-to-camera poses for every view.
3. **UV unwrapping** — uses [xatlas](https://github.com/jpcy/xatlas) to pack the mesh into a square texture atlas with no overlapping UVs.
4. **Back-projection** — for every texel in the atlas, finds the 3D surface point, projects it into all cameras, picks the best view (smallest positive depth = closest camera), and samples the colour bilinearly. Empty texels are filled by OpenCV inpainting.

Optional steps: **colour matching** across views before projection, and **midpoint subdivision** to increase mesh resolution after UV unwrapping.

## Installation

```bash
git clone https://github.com/photomesh/photomesh.git
cd photomesh
pip install -e .
```

For colour matching support, install the optional dependency:

```bash
pip install color-matcher
```

## Dataset Format

PhotoMesh reads a **COLMAP** dataset directory. The expected layout is:

```
dataset/
├── cameras.txt       # COLMAP cameras file (PINHOLE model)
├── images.txt        # COLMAP images file (poses)
├── images/           # image files referenced in images.txt
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
└── poisson.ply       # bare mesh to texture
```

> Only the **PINHOLE** camera model is currently supported. If your COLMAP reconstruction used a different model, undistort the images first (`colmap image_undistorter`).

If the images on disk have a different resolution than the one recorded in `cameras.txt` (e.g. you are using higher-resolution originals), PhotoMesh automatically rescales the intrinsics to match.

## Quick Start

```python
import photomesh

result = photomesh.map_texture(
    dataset_dir="dataset/",
    mesh_name="poisson.ply",
    output_dir="outputs/",   # auto-saves OBJ + PLY
)
```

Running this prints progress like:

```
[1/4] Loading mesh ...
  Mesh: 4766 verts, 9528 tris
[2/4] Loading COLMAP cameras ...
  9 cameras loaded
[3/4] UV-unwrapping ...
  UV: 5733 verts, 9528 tris
[4/4] Loading images & projecting ...
  Loaded 9 images
  Rasterising UV atlas (4096x4096) ...
  Inpainting 7077709 empty texels ...
  Saved outputs/textured.obj
  Saved outputs/textured.mtl
  Saved outputs/textured_texture.png
  Saved outputs/textured.ply
Done — per-texel back-projection at 4096x4096
```

### With colour matching

When images were captured under varying illumination, enabling colour matching reduces visible seams in the atlas:

```python
result = photomesh.map_texture(
    dataset_dir="dataset/",
    mesh_name="poisson.ply",
    color_match=True,          # transfer colour distribution to a common reference
    color_match_method="mkl",  # Monge-Kantorovich Linearization (recommended)
    color_match_ref=0,         # index of the reference image
    output_dir="outputs/",
)
```

### Accessing the result programmatically

```python
result = photomesh.map_texture(
    dataset_dir="dataset/",
    mesh_name="poisson.ply",
)

# Open3D mesh with vertex colours (sampled from the atlas)
mesh = result.get_open3d_mesh()
print(f"{len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

# Raw arrays
print(result.vertices.shape)   # (V, 3)  — seam-duplicated vertices
print(result.uvs.shape)        # (V, 2)  — normalised UV coords
print(result.atlas.shape)      # (H, W, 3) — uint8 RGB texture atlas

# Save manually with custom options
result.save("outputs/", prefix="my_mesh", formats=["obj"])
```

## API Reference

### `photomesh.map_texture()`

Main entry point. Orchestrates the full pipeline.

```python
photomesh.map_texture(
    dataset_dir: str,
    mesh_name: str = "poisson.ply",
    texture_size: int = 4096,
    uv_method: str | UVParametrizer = "xatlas",
    view_selector: str | ViewSelector = "closest_z",
    inpaint: bool = True,
    inpaint_radius: int = 3,
    color_match: bool = False,
    color_match_method: str = "mkl",
    color_match_ref: int = 0,
    subdivide: int = 0,
    output_dir: str | None = None,
    verbose: bool = True,
) -> TextureResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_dir` | `str` | — | Path to the COLMAP dataset directory |
| `mesh_name` | `str` | `"poisson.ply"` | Filename of the `.ply` mesh inside `dataset_dir` |
| `texture_size` | `int` | `4096` | Resolution of the square texture atlas |
| `uv_method` | `str` or `UVParametrizer` | `"xatlas"` | UV parametrization strategy |
| `view_selector` | `str` or `ViewSelector` | `"closest_z"` | View-selection strategy |
| `inpaint` | `bool` | `True` | Fill empty texels via OpenCV inpainting |
| `inpaint_radius` | `int` | `3` | Inpainting neighbourhood radius (pixels) |
| `color_match` | `bool` | `False` | Match colour / illumination across views before projection |
| `color_match_method` | `str` | `"mkl"` | Algorithm for colour matching (see below) |
| `color_match_ref` | `int` | `0` | Index of the reference image for colour matching |
| `subdivide` | `int` | `0` | Midpoint subdivision iterations (0 = none) |
| `output_dir` | `str` or `None` | `None` | If set, auto-saves results to this directory |
| `verbose` | `bool` | `True` | Print progress messages |

### `TextureResult`

Dataclass returned by `map_texture()`.

| Attribute | Type | Description |
|---|---|---|
| `vertices` | `np.ndarray (V, 3)` | Vertex positions (seam-duplicated by UV unwrap) |
| `triangles` | `np.ndarray (F, 3)` | Triangle indices into `vertices` |
| `uvs` | `np.ndarray (V, 2)` | Normalised UV coordinates in `[0, 1]` |
| `atlas` | `np.ndarray (H, W, 3)` | Baked texture atlas (uint8 RGB) |
| `mask` | `np.ndarray (H, W)` | Bool mask — `True` where a texel received colour |
| `texture_size` | `int` | Atlas resolution (e.g. 4096) |
| `mesh` | `o3d.geometry.TriangleMesh` | Open3D mesh with vertex colours sampled from the atlas |

**Methods:**

```python
result.save(
    output_dir: str = "outputs",
    prefix: str = "textured",
    formats: list[str] | None = None,   # default: ["obj", "ply"]
) -> list[str]   # returns paths of written files
```

Supported formats: `"obj"` (writes `.obj` + `.mtl` + `_texture.png`), `"ply"`.

```python
result.get_open3d_mesh() -> o3d.geometry.TriangleMesh
```

### `photomesh.match_colors()`

Standalone colour-matching utility (also called internally when `color_match=True`).

```python
photomesh.match_colors(
    images: list[np.ndarray],    # (H, W, 3) float32, values in [0, 255]
    reference_index: int = 0,
    method: str = "mkl",
) -> list[np.ndarray]
```

**Available methods:**

| Method | Description |
|---|---|
| `"mkl"` | Monge-Kantorovich Linearization — fast, high quality (recommended) |
| `"hm-mkl-hm"` | Histogram + MKL + Histogram compound — highest quality |
| `"hm"` | Classical histogram matching |
| `"reinhard"` | Reinhard et al. mean/std Lab transfer |
| `"default"` | Library default (currently `"hm-mkl-hm"`) |

Requires the `color-matcher` package (`pip install color-matcher`).

## Extending PhotoMesh

### Custom View Selector

Subclass `ViewSelector` and implement `select()`:

```python
from photomesh import ViewSelector
import numpy as np

class MySelector(ViewSelector):
    def select(
        self,
        pts3d: np.ndarray,    # (K, 3) 3D surface points
        cam_R: np.ndarray,    # (N, 3, 3) world-to-camera rotations
        cam_t: np.ndarray,    # (N, 3) world-to-camera translations
        cam_intr: np.ndarray, # (N, 6) [fx, fy, cx, cy, W, H]
        images: list,         # N loaded images as float32 (H, W, 3)
    ):
        # Return (best_colors, best_mask)
        # best_colors: (K, 3) float32 — sampled RGB per texel
        # best_mask: (K,) bool       — True where a colour was found
        ...

result = photomesh.map_texture(
    dataset_dir="dataset/",
    view_selector=MySelector(),
)
```

The built-in `ClosestZSelector` (string alias `"closest_z"`) picks the camera with the smallest positive depth for each texel, which generally produces the sharpest texture.

### Custom UV Parametrizer

Subclass `UVParametrizer` and implement `parametrize()`:

```python
from photomesh import UVParametrizer
import numpy as np

class MyUV(UVParametrizer):
    def parametrize(
        self,
        vertices: np.ndarray,  # (V, 3)
        triangles: np.ndarray, # (F, 3)
    ):
        # Return (new_verts, new_indices, new_uvs, vmapping)
        # new_verts:   (V', 3) — possibly seam-duplicated vertices
        # new_indices: (F, 3)  — triangle indices into new_verts
        # new_uvs:     (V', 2) — UV coords in [0, 1]
        # vmapping:    (V',)   — maps new vertex indices to original
        ...

result = photomesh.map_texture(
    dataset_dir="dataset/",
    uv_method=MyUV(),
)
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `open3d` | Mesh I/O and vertex-colour mesh |
| `Pillow` | Image loading |
| `opencv-python-headless` | Inpainting (`cv2.inpaint`) |
| `xatlas` | UV unwrapping (chart packing) |
| `color-matcher` *(optional)* | Colour/illumination matching across views |

## License

MIT
