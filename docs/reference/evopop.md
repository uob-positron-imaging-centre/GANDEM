# EvoPop API

The `evopop` module provides mesh post-processing and shape-factor-driven
optimisation for selecting particle subsets that match a target distribution.

Start with [Generate and process meshes](../guides/generate-and-process.md) or
[Select a representative particle population](../guides/select-population.md)
for complete workflows.

---

## Mesh pipeline

Build a sequential post-processing pipeline from composable steps:

```python
from gandem.evopop import (
    MeshPipeline,
    TopologyAware3DSegmenter,
    MeshRepair,
    ReduceTriangles,
    SmoothMesh,
)

pipeline = MeshPipeline([
    TopologyAware3DSegmenter(dem_safe=True),
    MeshRepair(),
    SmoothMesh(iterations=10),
    ReduceTriangles(target_reduction_frac=0.9),
])

mesh = pipeline.execute("particle.npy")
```

For predictable resource use, let `GeneratedParticlePool` select the worker count
automatically, or limit post-processing concurrency explicitly:

```python
from gandem.evopop import GeneratedParticlePool

pool = GeneratedParticlePool(
    n_samples=1_000,
    store_gen_dir="generated",
    mesh_pipeline=pipeline,
)
pool.generate(
    "generator.pt",
    batch_limit=None,        # estimate CUDA batch size; reduced on device OOM
    n_workers=4,             # post-processing worker count
)
```

### MeshPipeline

::: gandem.evopop.processing.MeshPipeline
    options:
      show_root_full_path: false

### MeshProcess

::: gandem.evopop.processing.MeshProcess
    options:
      show_root_full_path: false

### TopologyAware3DSegmenter

::: gandem.evopop.processing.TopologyAware3DSegmenter
    options:
      show_root_full_path: false

### MeshRepair

::: gandem.evopop.processing.MeshRepair
    options:
      show_root_full_path: false

### ReduceTriangles

::: gandem.evopop.processing.ReduceTriangles
    options:
      show_root_full_path: false

### SmoothMesh

::: gandem.evopop.processing.SmoothMesh
    options:
      show_root_full_path: false

### Otsu3DThreshold

::: gandem.evopop.processing.Otsu3DThreshold
    options:
      show_root_full_path: false

### SimpleThreshold

::: gandem.evopop.processing.SimpleThreshold
    options:
      show_root_full_path: false

---

## Shape optimisation

### GeneratedParticlePool

::: gandem.evopop.particle_selection.GeneratedParticlePool
    options:
      show_root_full_path: false

### ShapeOptimiser

::: gandem.evopop.particle_selection.ShapeOptimiser
    options:
      show_root_full_path: false

### estimate_target_distribution

::: gandem.evopop.particle_selection.estimate_target_distribution
    options:
      show_root_full_path: false

### select_best_epoch

::: gandem.evopop.particle_selection.select_best_epoch
    options:
      show_root_full_path: false

### OptimisationResult

::: gandem.evopop.particle_selection.OptimisationResult
    options:
      show_root_full_path: false

---

## Shape factor evaluation

### compute_shape_factors

::: gandem.evopop.shape_factors.compute_shape_factors
    options:
      show_root_full_path: false

### compute_shape_factors_3d

::: gandem.evopop.shape_factors.compute_shape_factors_3d
    options:
      show_root_full_path: false

### Projection2D

::: gandem.evopop.shape_factors.Projection2D
    options:
      show_root_full_path: false

### ShapeFactorResult

::: gandem.evopop.shape_factors.ShapeFactorResult
    options:
      show_root_full_path: false

### ShapeFactorBatch

::: gandem.evopop.shape_factors.ShapeFactorBatch
    options:
      show_root_full_path: false

### Low-level metrics

::: gandem.evopop.shape_factors.compute_sphericity
    options:
      show_root_full_path: false

::: gandem.evopop.shape_factors.compute_convexity
    options:
      show_root_full_path: false

::: gandem.evopop.shape_factors.compute_aspect_ratio
    options:
      show_root_full_path: false
