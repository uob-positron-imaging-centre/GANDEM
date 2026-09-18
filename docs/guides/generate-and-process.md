# Generate and process meshes

## Generate probability grids

Use a trusted generator checkpoint to create a batch of 3D occupancy
probabilities. `batch_limit=None` probes the accelerator and chooses a
conservative batch size (capped at 32); use an explicit value when you need
repeatable resource limits.

```python
from gandem.ganular import generate_from_checkpoint

generate_from_checkpoint(
    checkpoint_path="outputs/run-001/model_dump/generator_epoch_500.pth",
    n_samples=1_000,
    output_dir="generated/grids",
    dump_npy=True,
    dump_stl=False,
    batch_limit=None,
    threshold=0.5,
)
```

Use `dump_stl=True` only for a direct fixed-threshold conversion. For cleanup,
component selection, or DEM-oriented processing, retain `.npy` grids and use a
`MeshPipeline`.

## Build a valid pipeline

The first pipeline process must be a `GridProcess`, which changes a 3D
probability grid into a `trimesh.Trimesh`. Every following process must be a
`MeshProcess`.

```python
from gandem.evopop import (
    MeshPipeline,
    TopologyAware3DSegmenter,
    MeshRepair,
    SmoothMesh,
    ReduceTriangles,
)

pipeline = MeshPipeline([
    TopologyAware3DSegmenter(
        dem_safe=True,
        multiple_particles=False,
    ),
    MeshRepair(),
    SmoothMesh(iterations=10, lambda_=0.5),
    ReduceTriangles(target_reduction_frac=0.5),
])

pipeline.execute(
    "generated/grids/sample_0.npy",
    save_path="generated/meshes/sample_0.stl",
)
```

`dem_safe=True` requires `multiple_particles=False`. It is designed for a
single-particle output. Choose `SimpleThreshold(threshold=0.5)` when a direct,
predictable threshold is sufficient; choose `Otsu3DThreshold` when automatic
thresholding is appropriate. See the [EvoPop reference](../reference/evopop.md)
for their parameters.

## Generate and process a pool

`GeneratedParticlePool` connects generation with optional concurrent mesh
processing. It stores paths to resulting STL files for shape-factor work.

```python
from gandem.evopop import GeneratedParticlePool

pool = GeneratedParticlePool(
    n_samples=1_000,
    store_gen_dir="generated/pool",
    mesh_pipeline=pipeline,
)
pool.generate(
    checkpoint_path="outputs/run-001/model_dump/generator_epoch_500.pth",
    n_workers=4,
    timeout_seconds=120,
)
```

Leave `dump_npy=True` when supplying a mesh pipeline; the pipeline consumes the
generated `.npy` files. Failed per-grid worker jobs are warned about and do not
silently become valid population members—inspect warnings and the resulting
`pool.stl_files` count.

## Validate before simulation

1. Open representative STL outputs in a mesh viewer.
2. Check manifoldness, self-intersections, scale, and triangle count according
   to the target solver's requirements.
3. Compute both 2D projection and 3D factors when shape preservation matters.
   `compute_shape_factors_3d` requires watertight volume meshes.
4. Run an actual solver validation. A cleanup pipeline does not prove that
   contact behavior or maximum-overlap settings are appropriate.
