# Train, resume, and assess GANular

## Start a new run

The CLI exposes the training parameters most useful for routine experiments:

```bash
gandem-train cropped_particles \
  --output-dir outputs/run-001 \
  --epochs 500 \
  --object-batch-size 2 \
  --views-per-object 2 \
  --updates-per-epoch 100 \
  --seed 42
```

`--output-dir` must not already exist for a new run. Without it, GANular uses
`outputs/GANular_run`, which also must be absent. `--batch-size` aliases
`--object-batch-size`; `--views` aliases `--views-per-object`.

`epochs` means generator-training epochs. Each epoch contains exactly
`updates_per_epoch` generator updates. Increasing `views_per_object` raises the
projection batch size (`object_batch_size × views_per_object`) but does not
change the epoch length.

## Use the Python API for full configuration

```python
from gandem.ganular import QualityChecks, train

train(
    data_dir="cropped_particles",
    output_dir="outputs/quality-checked",
    epochs=500,
    object_batch_size=2,
    views_per_object=2,
    updates_per_epoch=100,
    critic_iterations=5,
    lambda_gp=10,
    lambda_binary=0.1,
    seed=42,
    quality_checks=QualityChecks(
        projection_reference_dir="images/held-out",
        uct_data_dir="measurements/reference-stls",
        evaluation_interval=10,
        sample_size=32,
        projection_sample_size=32,
    ),
)
```

The default 64 × 64 image resolution and model dimensions live in
`gandem.ganular.params`. Changing module-level parameters is an advanced,
code-level configuration change: model and projection dimensions must remain
compatible, and image resolution must be divisible by 16.

## Resume safely

Resume only from a full training checkpoint named `training_epoch_*.pth`.
GANular restores generator and discriminator weights, optimizer state, random
state, and training history. `epochs` becomes the number of **additional**
epochs.

```python
from gandem.ganular import train

train(
    resume_from="outputs/run-001/model_dump/training_epoch_500.pth",
    epochs=100,
)
```

GANular infers the original output directory unless you supply one, and refuses
to overwrite a later checkpoint. A generator-only checkpoint instead warm-starts
weights with a fresh optimizer state; it cannot reproduce an interrupted run.

## Read outputs and diagnostics

| Location | Meaning |
| --- | --- |
| `hyperparameters.json` | Effective run configuration and selected device |
| `model_dump/generator_epoch_*.pth` | Generator weights for inference |
| `model_dump/training_epoch_*.pth` | Full checkpoints for resuming |
| `gan_npy/` | Generated probability grids |
| `plots/` | Generator and discriminator loss history |
| `quality_reports/training_health.jsonl` | Per-step optimization-health records |
| `quality_reports/sample_quality.jsonl` | Periodic generated-sample diagnostics |

Quality checks are enabled by default. They record training health, 2D
projection agreement, raw topology, and—when `uct_data_dir` contains STL
meshes—3D fidelity and coverage. Their thresholds are advisory and should be
calibrated against a real-versus-real reference split. To disable them from the
CLI, pass `--no-quality-checks`.

## Diagnose common failures

| Symptom | Cause and action |
| --- | --- |
| `No images found` | Put supported image files directly in `data_dir`; subdirectories are not scanned. |
| `FileExistsError` at startup | Pick a new output directory; do not reuse a prior run directory. |
| GPU memory error | Reduce `object_batch_size` or `views_per_object`; both affect projection-batch memory. |
| Poor or saturated projections | Inspect input polarity and segmentation first. The loader expects dark particles on a light background. |
| Quality checks fail | Treat the report as a diagnostic. Check image quality, reference data, generated grids, and mesh post-processing before tuning thresholds. |

Never load a checkpoint from an untrusted source.
