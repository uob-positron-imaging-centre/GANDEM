# GANular API

The `ganular` module implements a projection-based WGAN-GP for learning to
generate 3D voxel representations of particles from 2D projection images.

Use [Train, resume, and assess GANular](../guides/training.md) for the
operational workflow and [the CLI reference](cli.md) for installed command
options.

## Quality checks

Training records five advisory checks under ``<output_dir>/quality_reports``:

1. training health from losses, gradient norms, and projection saturation;
2. 2D agreement using circularity, aspect ratio, and convexity;
3. raw topology using boundary contact, component count, watertightness,
   winding consistency, volume validity, and genus;
4. raw 3D fidelity using 3D convexity, sphericity, and aspect ratio; and
5. raw 3D coverage in the same three-feature space.

The topology check works without reference uCT data. The fidelity and coverage
checks require segmented uCT surfaces stored as one ``.stl`` file per particle.
Reference meshes are measured directly without resampling or mesh repair;
generated voxel grids are thresholded once at the configured fixed threshold.

```python
from gandem.ganular import QualityChecks, train

train(
    data_dir="cropped_particles/MCC_Vivapur_102",
    output_dir="outputs/quality_checked_run",
    quality_checks=QualityChecks(
        uct_data_dir="uct_particles_stl",
        projection_reference_dir="held_out_projections",
        evaluation_interval=10,
        sample_size=32,
        projection_sample_size=32,
        voxel_threshold=0.5,
    ),
)
```

Training treats generated objects and their projections as separate statistical
units. Each update generates ``object_batch_size`` particles, renders
``views_per_object`` views of each, and samples the resulting
``object_batch_size * views_per_object`` real projections independently.
``updates_per_epoch`` fixes epoch length when either batch dimension changes.

``training_health.jsonl`` contains the per-step training-health check. The less
frequent ``sample_quality.jsonl`` contains all five results and their raw
topology, distance, fidelity, and coverage values. Thresholds are intentionally
configurable and should be calibrated against real-versus-real uCT splits.

---

## Training

To continue a run, pass its full training checkpoint. ``epochs`` is the number
of additional epochs; when ``output_dir`` is omitted, the existing run directory
is inferred from the checkpoint path.

```python
train(
    resume_from="outputs/quality_checked_run/model_dump/training_epoch_500.pth",
    epochs=100,
)
```

::: gandem.ganular.train.train
    options:
      show_root_full_path: false

## Inference

::: gandem.ganular.model.generate_from_checkpoint
    options:
      show_root_full_path: false

## Configuration

::: gandem.ganular.params
    options:
      show_root_full_path: false
      show_if_no_docstring: true

## Models

### Generator

::: gandem.ganular.model.VoxelGenerator
    options:
      show_root_full_path: false
      members:
        - __init__
        - forward

### Discriminator (v2)

::: gandem.ganular.model.ProjectionDiscriminator
    options:
      show_root_full_path: false
      members:
        - __init__
        - forward
        - _block
        - initialize_weights

### Projection

::: gandem.ganular.model.VoxelProjector
    options:
      show_root_full_path: false
      members:
        - __init__
        - forward
        - get_rotation_mat

### Straight-Through Estimator

::: gandem.ganular.model.BinaryThresholdSTE
    options:
      show_root_full_path: false

## Datasets

::: gandem.ganular.datasets.ProjectionDataset
    options:
      show_root_full_path: false

## Utilities

::: gandem.ganular.artifacts.save_voxel_stl
    options:
      show_root_full_path: false

::: gandem.ganular.artifacts.save_voxel_npy
    options:
      show_root_full_path: false

::: gandem.ganular.artifacts.save_voxel_plot
    options:
      show_root_full_path: false

::: gandem.ganular.artifacts.save_loss_plot
    options:
      show_root_full_path: false

::: gandem.ganular.artifacts.save_projection_plot
    options:
      show_root_full_path: false

::: gandem.ganular.train.compute_gradient_penalty
    options:
      show_root_full_path: false
