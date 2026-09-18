# Install and verify GAN-DEM

## Requirements

- Python 3.10 or newer.
- A current PyTorch-supported accelerator for practical training. GANular
  chooses the current accelerator when available, otherwise CPU.
- Sufficient disk space for images, checkpoints, voxel grids, and STL meshes.
  These artifacts can grow quickly with sample count.

Mesh repair relies on PyVista and PyMeshFix. On a headless Linux machine, a
working off-screen graphics stack may be required by the local PyVista build.

## Install from this repository

```bash
git clone https://github.com/abhirup-roy/GAN-DEM.git
cd GAN-DEM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[docs]"
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Use `.[dev]` instead of `.[docs]` when you also need the development test and
formatting dependencies. Combine extras as `.[dev,docs]`.

## Verify the installation

```bash
python -c "import gandem; print(gandem.__all__)"
gandem-train --help
```

The second command confirms that the package's supported command-line entry
point is available. If it is not found, make sure the virtual environment is
active and reinstall with `python -m pip install -e .`.

## Create a project layout

Keep inputs and generated artifacts outside the source tree:

```text
particle-study/
├── images/raw/                 # Original silhouettes or projections
├── cropped_particles/          # Default output of crop_particles()
├── outputs/run-001/            # Training output directory
├── generated/                  # Generated grids and processed meshes
└── measurements/               # Real shape-factor data or reference STLs
```

New GANular runs refuse to reuse an existing output directory. Choose a unique
run directory, rather than deleting or overwriting old results.

## Prepare suitable training images

GANular scans only the top-level training directory for `.jpg`, `.jpeg`,
`.png`, `.bmp`, `.tif`, and `.tiff` files. It reads each image as grayscale,
inverts it, resizes it to 64 × 64 pixels using nearest-neighbour interpolation,
then applies a 0.5 binary threshold.

Use isolated silhouettes with a dark particle on a light background. Exclude
scale bars, labels, edge-clipped particles, and multi-particle regions. Use
[`crop_particles`](reference/pre.md#particle-cropping) to create a flat output
directory from `.jpg` silhouettes or projections; it writes to `cropped_particles/` in
the current working directory.

## Reproducibility checklist

- Set `seed` for a run when repeatability matters.
- Preserve the source-image directory, `hyperparameters.json`, and full
  `training_epoch_*.pth` checkpoints.
- Record the package revision and accelerator used.
- Do not load a checkpoint from an untrusted source; PyTorch checkpoints use
  Python pickle internally.
