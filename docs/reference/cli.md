# Command-line reference

GAN-DEM provides one installed command: `gandem-train`.

```text
gandem-train DATA_DIR [options]
```

`DATA_DIR` is a directory containing supported particle image files directly
inside it.

| Option | Meaning |
| --- | --- |
| `-o`, `--output-dir PATH` | New directory for run outputs. Must not exist for a new run. |
| `--epochs N` | Training epochs; positive integer. |
| `--object-batch-size N`, `--batch-size N` | Independently generated objects per update. |
| `--views-per-object N`, `--views N` | Rendered projection views per generated object. |
| `--updates-per-epoch N` | Generator updates in each epoch. |
| `--projection-sample-size N` | Fixed number of 2D reference projections at each quality evaluation. |
| `--seed N` | Random seed. |
| `--resume-from PATH` | Full `training_epoch_*.pth` checkpoint to continue. |
| `--no-quality-checks` | Disable advisory quality diagnostics. |

Run `gandem-train --help` to view the installed command's help text. Use the
Python API for `QualityChecks`, learning rates, critic iterations, loss weights,
uCT reference data, inference, and all mesh-processing workflows.
