# Pre-processing API

`gandem.pre` prepares 2D training silhouettes or projections by extracting
particles from supplied silhouettes or projections, generating simple synthetic
shapes, or projecting STL meshes. For a workflow guide, see [Install and verify
GAN-DEM](../getting-started.md).

---

## Particle cropping

::: gandem.pre.segregation.crop_particles
    options:
      show_root_full_path: false

::: gandem.pre.segregation.process_single_image
    options:
      show_root_full_path: false

::: gandem.pre.segregation.find_best_max_size
    options:
      show_root_full_path: false

---

## Synthetic image generation

Generate simple geometric shapes for testing and validation:

```python
from gandem.pre import generate_pentagons, generate_diamonds

generate_pentagons(n=500, output_dir="test_data/pentagons", random_skew=True)
generate_diamonds(n=500, output_dir="test_data/diamonds", random_skew=True)
```

::: gandem.pre.synthetic_shapes_2d.generate_pentagons2d
    options:
      show_root_full_path: false

::: gandem.pre.synthetic_shapes_2d.generate_diamonds2d
    options:
      show_root_full_path: false

---

## STL to projection conversion

Convert existing 3D meshes into 2D projection images suitable for training:

```python
from pathlib import Path
from gandem.pre import convert_stls_to_projections

stl_files = list(Path("meshes").glob("*.stl"))
convert_stls_to_projections(
    stl_files, output_dir="projections", auto_scale=True
)
```

::: gandem.pre.stl_projection.convert_stls_to_projections
    options:
      show_root_full_path: false

::: gandem.pre.stl_projection.compute_max_grid_size
    options:
      show_root_full_path: false
