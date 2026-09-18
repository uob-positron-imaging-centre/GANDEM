# GAN-DEM

GAN-DEM is a Python toolkit for building 3D particle populations from 2D
silhouettes for discrete element method (DEM) studies. Its workflow is:

```text
2D particle images → GANular training → voxel probability grids
                                            ↓
                         EvoPop segmentation and mesh repair → STL meshes
                                                                    ↓
                                           shape-distribution selection → DEM input set
```

The package is research software. Generated geometry and its distribution need
validation in the intended DEM solver; GAN-DEM does not certify mechanical or
contact-model suitability.

## Choose a path

| If you want to… | Start here |
| --- | --- |
| Run the full workflow for the first time | [End-to-end tutorial](tutorials/end-to-end.md) |
| Install and check the environment | [Installation](getting-started.md) |
| Train, resume, or diagnose GANular | [Training GANular](guides/training.md) |
| Turn generated grids into usable STL meshes | [Generate and process meshes](guides/generate-and-process.md) |
| Match a selected population to measured particle data | [Select a particle population](guides/select-population.md) |
| Look up a function, class, or CLI option | [API reference](reference/cli.md) |

## Components

| Component | Responsibility | Main inputs | Main outputs |
| --- | --- | --- | --- |
| [`pre`](reference/pre.md) | Crop, synthesize, and project training silhouettes | Images or STL meshes | 2D image files |
| [`ganular`](reference/ganular.md) | Learn a 3D voxel generator from 2D projections | A flat image directory | Checkpoints, voxel grids, diagnostics |
| [`evopop`](reference/evopop.md) | Segment, repair, characterize, and select meshes | `.npy` probability grids or STL files | STL meshes and selected files |

## Core assumptions

- Training images are loaded in grayscale, inverted, resized to 64 × 64, and
  binarized. The expected foreground is therefore a dark particle on a light
  background before loading.
- The GAN learns from 2D projections rather than paired 2D/3D examples. It
  produces 3D occupancy probabilities, not physical dimensions or material
  properties.
- Mesh processing begins with a probability-grid-to-mesh step. A
  `MeshPipeline` must therefore begin with a `GridProcess`, such as
  `TopologyAware3DSegmenter` or `SimpleThreshold`.
- Shape selection compares geometric descriptors. It cannot replace
  solver-specific validation of contacts, overlaps, or material behavior.

## Documentation conventions

Paths in examples are relative to the project directory. Examples use Python
APIs when a workflow requires configuration beyond the CLI. Names ending in
`.pth` or `.pt` refer to PyTorch checkpoints; load them only when their origin
is trusted.
