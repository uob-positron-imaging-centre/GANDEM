# Supporting utilities API

These lower-level functions support synthetic-data experiments and custom
projection workflows. They are not required for the standard
train–generate–process path.

## 3D synthetic shapes

`generate_diamonds3d` creates randomly scaled, rotated octahedral STL meshes.
Use it to make controlled test data, not as a substitute for measured particle
geometry.

```python
from gandem.pre.synthetic_shapes_3d import generate_diamonds3d, generate_projections

generate_diamonds3d(100, "synthetic/stls", random_skew=True)
generate_projections(500, "synthetic/stls", "synthetic/projections")
```

`generate_projections` samples STL files with random rotations and writes
inverted PNG silhouettes, compatible with GANular's input-polarity convention.

::: gandem.pre.synthetic_shapes_3d.generate_diamonds3d
    options:
      show_root_full_path: false

::: gandem.pre.synthetic_shapes_3d.generate_projections
    options:
      show_root_full_path: false

## Mesh-outline projection

`project_onto_axes` rasterizes mesh outlines onto selected coordinate planes.
It expects a mesh object with triangle geometry and returns one `(n, 2)` outline
array per requested axis. This is useful for custom measurement workflows;
EvoPop's shape-factor APIs normally handle projection internally.

::: gandem.mesh_projection.project_onto_axes
    options:
      show_root_full_path: false
