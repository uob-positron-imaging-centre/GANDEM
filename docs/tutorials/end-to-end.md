# End-to-end tutorial

This tutorial takes a directory of input silhouettes or projections through a minimal
train–generate–mesh workflow. It assumes GAN-DEM is [installed](../getting-started.md)
and the environment is active.

## 1. Crop particles

Place `.jpg/.png` projection images in `images/raw`. Crop individual particles to
the `cropped_particles/` directory created in the current working directory.
`max_size` is the minimum accepted contour area in pixels; choose it to exclude
small image artifacts.

```python
from gandem.pre import crop_particles

crop_particles(
    img_dir="images/raw",
    max_size=50,
)
```

Review the generated images before training. Edge-clipped particles are
skipped and reported as a warning. Confirm that particles are dark and the
background is light, because the dataset loader inverts them.

## 2. Train GANular

Start with a small run to confirm the data and output layout. The product of
`object_batch_size` and `views_per_object` is the number of real projections
sampled for each critic update.

```bash
gandem-train cropped_particles \
  --output-dir outputs/tutorial-run \
  --epochs 5 \
  --object-batch-size 2 \
  --views-per-object 2 \
  --updates-per-epoch 10 \
  --seed 42
```

Inspect `outputs/tutorial-run/plots`, `gan_projections`, and `gan_gens` before
committing to a long run. A normal run also creates:

```text
outputs/tutorial-run/
├── hyperparameters.json
├── model_dump/                 # generator, discriminator, and full training checkpoints
├── gan_npy/                    # generated 3D probability grids
├── gan_gens/                   # voxel previews
├── gan_projections/            # projected generated samples
├── plots/                      # loss plots
└── quality_reports/            # advisory JSON/JSONL diagnostics
```

## 3. Generate a pool from a checkpoint

Use a generator checkpoint for inference. A `generator_epoch_*.pth` checkpoint
is enough to generate; a `training_epoch_*.pth` checkpoint is required to
resume training.

```python
from gandem.ganular import generate_from_checkpoint

generate_from_checkpoint(
    checkpoint_path="outputs/tutorial-run/model_dump/generator_epoch_5.pth",
    n_samples=100,
    output_dir="generated/grids",
    dump_npy=True,
    dump_stl=False,
    threshold=0.5,
)
```

The generated `.npy` files contain 3D probabilities in the interval [0, 1].
Keep these grids when you plan to use topology-aware segmentation; converting
them straight to STL uses a fixed threshold only.

## 4. Create cleaned STL meshes

Build a pipeline whose first element converts the probability grid to a mesh.
The following configuration selects and cleans a primary component, repairs it,
smooths it, and reduces triangle count.

```python
from gandem.evopop import (
    MeshPipeline,
    MeshRepair,
    ReduceTriangles,
    SmoothMesh,
    TopologyAware3DSegmenter,
)

pipeline = MeshPipeline([
    TopologyAware3DSegmenter(dem_safe=True),
    MeshRepair(),
    SmoothMesh(iterations=10),
    ReduceTriangles(target_reduction_frac=0.5),
])

mesh = pipeline.execute(
    "generated/grids/sample_0.npy",
    save_path="generated/meshes/sample_0.stl",
)
```

Review the output in the target mesh viewer and run solver-specific checks.
Mesh validity, triangle count, and shape statistics are distinct requirements.

## Next steps

- Use [Training GANular](../guides/training.md) to resume a run and interpret
  advisory quality reports.
- Use [Generate and process meshes](../guides/generate-and-process.md) to
  process an entire pool safely.
- Use [Select a particle population](../guides/select-population.md) to match
  generated particles to a measured shape distribution.
