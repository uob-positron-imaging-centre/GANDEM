# Select a representative particle population

Use EvoPop after you have a pool of valid STL files and measurements describing
the target population. The optimizer selects `n_select` unique mesh IDs and,
optionally, their mixture frequencies.

## Build the target and generated descriptors

The selection workflow compares three factors: sphericity, convexity, and
aspect ratio. `estimate_target_distribution` accepts a one-factor sample or a
three-factor sample array and returns sampled target observations.

```python
from pathlib import Path

import numpy as np
from gandem.evopop import (
    GeneratedParticlePool,
    ShapeOptimiser,
    estimate_target_distribution,
)

# Shape (n_observations, 3), in sphericity/convexity/aspect-ratio order.
real_measurements = np.load("measurements/real_shape_factors.npy")
target = estimate_target_distribution(real_measurements, n=200, seed=42)

pool = GeneratedParticlePool.from_stls(list(Path("generated/meshes").glob("*.stl")))
pool.populate_shape_factors(kind="outline")
```

For STL files, `kind="outline"` is required. Multiple rotations or views add
observations per particle; they do not add selectable particle IDs.

## Optimize and export a subset

```python
optimiser = ShapeOptimiser(
    optimal_distribution=target,
    generations_pool=pool,
    n_select=10,
    max_passes=100,
    distance_metric="wasserstein",
)
result = optimiser.optimise(seed=42, progress=True)

print(result.best_params)        # selected STL IDs
print(result.best_frequencies)   # their mixture frequencies
print(optimiser.ks_validate())
optimiser.copy_selected("generated/selected")
```

Use `distance_metric="wasserstein"` to compare each factor's marginal
distribution, or `"energy"` to compare the joint three-dimensional
distribution and capture dependencies between factors. `result.history` is the
configured objective; per-factor diagnostics remain Wasserstein distances in
their native units.

## Interpret results carefully

- Use `initialisation="stratified"` (the default) for a broad starting subset.
- `factor_scales=None` preserves native factor units. Do not set
  `factor_scales="std"` automatically: compare against random subsets first.
- `polish=True` improves the scalar 2D objective with local swaps. Disable it
  when preserving 3D fidelity is more important than the final increment of
  2D fit.
- `ks_validate()` gives a two-sample statistical check per factor, not proof
  that the selected meshes are physically equivalent to measured particles.

For a direct array-based workflow, pass a finite `(3, observations)` or
`(4, observations)` descriptor matrix to `ShapeOptimiser`. The fourth row,
when present, is the particle ID.
