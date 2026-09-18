"""Select STL meshes whose 2D shape-factor distributions match a target."""

from __future__ import annotations
import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass
import signal
import time
import queue
import traceback
from collections import deque
from os import PathLike
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Generic, NamedTuple, Optional, Sequence, TypeVar
from warnings import warn


from cma import CMAEvolutionStrategy
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist
import torch
from tqdm import tqdm, trange

from ..ganular.model import generate_from_checkpoint
from .shape_factors import common_mesh_scale, compute_shape_factors
from .processing import MeshPipeline

SHAPE_FACTORS = ("sphericity", "aspect_ratio", "convexity")
INITIALISATIONS = ("stratified", "greedy", "neutral")
DISTANCE_METRICS = ("wasserstein", "energy")

# The random-key encoding makes the cost a step function of the search
# coordinates: every candidate decoding to the same subset scores identically,
# which happens routinely once the population concentrates. pycma's default
# tolflatfitness of 1 therefore ends a run after two such generations, long
# before it has converged, so a sustained run of flat generations is required
# instead.
FLAT_FITNESS_TOLERANCE = 20
ParamsT = TypeVar("ParamsT")


@contextmanager
def _isolated_global_random_state():
    """Keep pycma's reseeding out of the caller's global random stream.

    pycma offers no local generator: it calls ``np.random.seed()`` while
    constructing a strategy and then draws every candidate from the
    process-wide legacy generator. A seeded run would therefore reseed global
    numpy for everything downstream -- including the projection axes that
    :mod:`.shape_factors` draws with ``np.random.choice`` -- which silently
    couples a selection run to the measurements taken after it.

    Nothing inside a run reads the global generator, so the caller's state can
    simply be handed back untouched. Unseeded runs stay independent of one
    another because pycma derives its clock seed after calling
    ``np.random.seed()`` with fresh entropy, not from the incoming state.
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class KSResults(NamedTuple):
    sphericity: bool
    aspect_ratio: bool
    convexity: bool


def _execute_mesh_pipeline(mesh_pipeline: MeshPipeline, grid: PathLike, result) -> None:
    """Run one pipeline in a disposable child and return a small result."""
    try:
        result.put(("ok", mesh_pipeline.execute(grid) is not None))
    except (TimeoutError, RuntimeError) as exc:
        result.put(("ok", False))
        warn(f"Mesh pipeline failed for '{grid}': {exc}")
    except BaseException:
        result.put(("error", traceback.format_exc()))


def _describe_exit(process) -> str:
    """Explain why a worker ended without publishing a result."""
    code = process.exitcode
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return (
            f"was killed by {name} (exit code {code}) before returning a result; "
            "the operating system, job scheduler, or an out-of-memory guard "
            "stopped it rather than the pipeline itself failing"
        )
    return f"exited with code {code} without returning a result"


def _run_mesh_pipelines(
    mesh_pipeline: MeshPipeline,
    grids: Sequence[PathLike],
    *,
    workers: int,
    timeout_seconds: float,
    progress: bool,
    context=None,
) -> tuple[int, list[Path], list[Path]]:
    """Run each grid in an independently killable process."""
    context = context or mp.get_context("spawn")
    pending = deque(Path(path) for path in grids)
    active = {}
    discarded = 0
    timed_out = []
    killed = []
    progress_bar = tqdm(
        total=len(pending),
        desc="Post-processing meshes",
        unit="mesh",
        disable=not progress,
    )

    try:
        while pending or active:
            while pending and len(active) < workers:
                path = pending.popleft()
                result = context.Queue()
                process = context.Process(
                    target=_execute_mesh_pipeline,
                    args=(mesh_pipeline, path, result),
                )
                process.start()
                active[path] = (process, result, time.monotonic())

            for path, (process, result, started) in list(active.items()):
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join()
                    else:
                        process.join()
                    path.with_suffix(".stl").unlink(missing_ok=True)
                    discarded += 1
                    timed_out.append(path)
                elif process.is_alive():
                    continue
                else:
                    process.join()
                    try:
                        status, payload = result.get(timeout=1)
                    except queue.Empty:
                        status = "killed"
                        payload = _describe_exit(process)
                    if status == "error":
                        raise RuntimeError(
                            f"Mesh pipeline failed for '{path}':\n{payload}"
                        )
                    if status == "killed":
                        warn(f"Mesh pipeline worker for '{path}' {payload}")
                        killed.append(path)
                    if status == "killed" or not payload:
                        path.with_suffix(".stl").unlink(missing_ok=True)
                        discarded += 1

                result.close()
                del active[path]
                progress_bar.update()

            if active:
                time.sleep(0.05)
    finally:
        for process, result, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            result.close()
        progress_bar.close()

    return discarded, timed_out, killed


def _as_sample_rows(data: np.ndarray) -> np.ndarray:
    """Return shape-factor data as ``(samples, factors)``."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("shape-factor data must be a one- or two-dimensional array")
    if data.shape[1] not in (1, 3) and data.shape[0] in (1, 3):
        data = data.T
    if data.shape[1] not in (1, 3):
        raise ValueError("shape-factor data must contain one or three factors")
    if not np.all(np.isfinite(data)):
        raise ValueError("shape-factor data must contain only finite values")
    return data


def target_factor_scales(target: np.ndarray) -> np.ndarray:
    """Per-factor standard deviation of a target distribution.

    A Wasserstein distance carries the units of its own variable, so weighting
    the three raw distances ranks the factors by the width of their axis
    rather than by the weights supplied. Dividing by these scales makes equal
    ``factor_weights`` mean equal priority.

    Warning:
        This is rarely the right thing to do for shape factors, and it is not
        the default. Convexity occupies a much narrower range than aspect
        ratio, so normalising promotes convexity to the *most* important
        objective -- and convexity is usually the factor carrying the least
        real signal and the most sampling noise. On representative synthetic
        pools, normalising drove convexity well below the distance a random
        subset of the same size achieves, which is overfitting, and paid for
        it by making aspect ratio roughly twice as bad as random. Prefer
        explicit ``factor_weights`` on the raw distances, or scales derived
        from each factor's attainable range, and always compare a result
        against a random subset of the same size before adopting it.

    Args:
        target: Target observations shaped ``(3, samples)``.

    Returns:
        Standard deviation of each factor, floored so a degenerate factor
        cannot divide the cost by zero.
    """
    scales = np.std(np.asarray(target, dtype=float), axis=1)
    return np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)


def estimate_target_distribution(
    X_real: np.ndarray,
    n: int,
    distribution: str = "kde",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a target distribution fitted to empirical 2D measurements.

    Args:
        X_real: Empirical measurements shaped ``(samples, 3)``. A transposed
            ``(3, samples)`` array and single-factor data are also accepted.
        n: Number of target observations to draw.
        distribution: ``"kde"``, ``"empirical"``, ``"normal"``, or
            ``"lognormal"``. Normal and lognormal distributions are fitted
            independently to each shape factor.
        seed: Random seed.

    Returns:
        Target observations shaped ``(factors, n)``, ready for
        :class:`ShapeOptimiser`.

    Notes:
        Shape factors are bounded by zero and one. KDE uses a logit transform;
        parametric samples outside the empirical factor bounds are clipped.
    """
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")

    data = _as_sample_rows(X_real)
    name = distribution.lower()
    rng = np.random.default_rng(seed)

    if name == "empirical":
        samples = data[rng.integers(0, len(data), size=n)]
    elif name == "kde":
        if len(data) < 2:
            raise ValueError("KDE requires at least two empirical observations")
        eps = np.finfo(float).eps
        bounded = np.clip(data, eps, 1 - eps)
        transformed = np.log(bounded / (1 - bounded))
        try:
            samples = stats.gaussian_kde(transformed.T).resample(n, seed=rng).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("KDE requires non-degenerate empirical data") from exc
        samples = 1 / (1 + np.exp(-samples))
    elif name in ("normal", "norm", "lognormal", "lognorm"):
        columns = []
        for values in data.T:
            if name in ("normal", "norm"):
                loc, scale = stats.norm.fit(values)
                drawn = stats.norm.rvs(loc=loc, scale=scale, size=n, random_state=rng)
            else:
                if np.any(values <= 0):
                    raise ValueError(
                        "lognormal fitting requires strictly positive data"
                    )
                shape, loc, scale = stats.lognorm.fit(values, floc=0)
                drawn = stats.lognorm.rvs(
                    shape, loc=loc, scale=scale, size=n, random_state=rng
                )
            columns.append(np.clip(drawn, values.min(), values.max()))
        samples = np.column_stack(columns)
    else:
        raise ValueError(
            "distribution must be 'kde', 'empirical', 'normal', or 'lognormal'"
        )

    return samples.T


class GeneratedParticlePool:
    """STL candidates and all shape-factor observations from their projections.

    Existing STL files can be supplied directly with :meth:`from_stls`. The
    original generation workflow is retained for callers that want this class
    to generate and post-process a new candidate pool.
    """

    def __init__(
        self,
        n_samples: Optional[int] = None,
        store_gen_dir: Optional[PathLike] = None,
        mesh_pipeline: Optional[MeshPipeline] = None,
        stl_files: Optional[Sequence[PathLike]] = None,
    ):
        self.stl_files = (
            [Path(path) for path in stl_files] if stl_files is not None else []
        )
        if self.stl_files:
            missing = [str(path) for path in self.stl_files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"STL file not found: {missing[0]}")
            invalid = [
                str(path) for path in self.stl_files if path.suffix.lower() != ".stl"
            ]
            if invalid:
                raise ValueError(f"Expected an STL file: {invalid[0]}")

        self.n_samples = (
            int(n_samples) if n_samples is not None else len(self.stl_files)
        )
        if store_gen_dir is None:
            store_gen_dir = self.stl_files[0].parent if self.stl_files else "."
        self.store_gen_dir = Path(store_gen_dir)
        self.store_gen_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_pipeline = mesh_pipeline
        self.shape_factors: Optional[np.ndarray] = None

    @classmethod
    def from_stls(cls, stl_files: Sequence[PathLike]) -> "GeneratedParticlePool":
        """Create a candidate pool from processed STL files."""
        return cls(stl_files=stl_files)

    def generate(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        out_filename: str = "sample",
        dump_npy: bool = True,
        batch_limit: Optional[int] = None,
        n_workers: Optional[int] = None,
        timeout_seconds: float = 300.0,
        threshold: float = 0.5,
        progress: bool = True,
    ) -> None:
        """Generate and optionally post-process a new pool of particles.

        Generated STL paths are available from :attr:`stl_files` when this
        method returns. If a mesh pipeline is configured, ``dump_npy`` must be
        enabled because the pipeline consumes probability grids. The pipeline
        timeout applies independently to each generated grid.
        """
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive before generating a pool")
        if n_workers is not None and n_workers < 1:
            raise ValueError("n_workers must be a positive integer")
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.mesh_pipeline and not dump_npy:
            warn("mesh_pipeline is configured but dump_npy=False; skipping it")
        generate_from_checkpoint(
            checkpoint_path=checkpoint_path,
            n_samples=self.n_samples,
            output_dir=self.store_gen_dir,
            device=device,
            out_filename=out_filename,
            dump_npy=dump_npy,
            batch_limit=batch_limit,
            threshold=threshold,
            progress=progress,
        )

        if self.mesh_pipeline and dump_npy:
            npy_files = sorted(self.store_gen_dir.glob("*.npy"))
            if not npy_files:
                warn("No .npy probability grids were generated; skipping mesh pipeline")
                self.stl_files = sorted(self.store_gen_dir.glob("*.stl"))
                return
            workers = min(n_workers or mp.cpu_count(), len(npy_files))
            discarded, timed_out, killed = _run_mesh_pipelines(
                self.mesh_pipeline,
                npy_files,
                workers=workers,
                timeout_seconds=float(timeout_seconds),
                progress=progress,
            )
            if killed:
                names = ", ".join(path.name for path in killed)
                warn(
                    f"{len(killed)} mesh pipeline workers were killed by the "
                    f"system and were rejected: {names}"
                )
            if timed_out:
                names = ", ".join(path.name for path in timed_out)
                warn(
                    f"{len(timed_out)} meshes exceeded the "
                    f"{timeout_seconds:g}-second timeout and were "
                    f"rejected: {names}"
                )
            if discarded:
                warn(
                    f"{discarded} of {self.n_samples} meshes specified for the pool "
                    "were discarded (removed from the pool) because processing did "
                    "not produce accepted meshes."
                )
        self.stl_files = sorted(self.store_gen_dir.glob("*.stl"))

    def mesh_scale(self) -> tuple[float, float]:
        """Measure the common scale to compare this pool against a target at.

        The pool is usually the coarser of the two populations, since its meshes
        have been through the mesh pipeline. Pass both returned values to
        :meth:`populate_shape_factors` and to the
        :func:`~.shape_factors.compute_shape_factors` call that measures the
        target, so both are measured at one scale::

            chord, density = pool.mesh_scale()
            pool.populate_shape_factors(
                outline_chord=chord, match_face_density=density
            )
            compute_shape_factors(
                target_stls, kind="outline",
                outline_chord=chord, match_face_density=density,
            )
        """
        if not self.stl_files:
            self.stl_files = sorted(self.store_gen_dir.glob("*.stl"))
        if not self.stl_files:
            raise ValueError("the generations pool contains no STL files")
        return common_mesh_scale(self.stl_files)

    def populate_shape_factors(
        self,
        kind: str = "outline",
        num_workers: Optional[int] = None,
        rotations_per_stl: int = 1,
        views_per_rotation: int = 3,
        outline_chord: Optional[float] = None,
        match_face_density: Optional[float] = None,
        seed: int = 0,
        progress: bool = True,
    ) -> np.ndarray:
        """Measure every requested projection without averaging by STL.

        The returned array has rows for sphericity, aspect ratio, convexity,
        and zero-based STL ID. Keeping the ID row lets the optimiser select
        whole STL files while comparing the frequency of all their 2D views.

        Args:
            outline_chord: Arc-length step to re-sample every outline at.
            match_face_density: Faces per unit area to decimate each mesh to.
            seed: Seeds the projection orientations.

        A pool whose meshes have been decimated (e.g. by
        :class:`~.processing.ReduceTriangles`) projects a shorter perimeter
        than the raw ground truth, which inflates its 2D sphericity. Take both
        scale arguments from :meth:`mesh_scale` and pass them here *and* to the
        target's :func:`~.shape_factors.compute_shape_factors` call.
        """
        if kind != "outline":
            raise ValueError("STL projection shape factors require kind='outline'")
        if rotations_per_stl < 0:
            raise ValueError("rotations_per_stl must be non-negative")
        if not 1 <= views_per_rotation <= 3:
            raise ValueError("views_per_rotation must be between 1 and 3")

        if not self.stl_files:
            self.stl_files = sorted(self.store_gen_dir.glob("*.stl"))
        if not self.stl_files:
            raise ValueError("the generations pool contains no STL files")

        result = compute_shape_factors(
            images=self.stl_files,
            kind=kind,
            num_workers=num_workers,
            rotations_per_stl=rotations_per_stl,
            views_per_rotation=views_per_rotation,
            outline_chord=outline_chord,
            match_face_density=match_face_density,
            seed=seed,
            progress=progress,
        ).to_array()
        ids = result[3].astype(int)
        valid = np.all(np.isfinite(result[:3]), axis=0)
        self.shape_factors = np.vstack((result[:3, valid], ids[valid]))

        measured = set(self.shape_factors[3].astype(int))
        missing = sorted(set(range(len(self.stl_files))) - measured)
        if missing:
            raise ValueError(
                f"no valid 2D projections were produced for {self.stl_files[missing[0]]}"
            )
        return self.shape_factors


@dataclass
class OptimisationResult(Generic[ParamsT]):
    """Optimization history with its best parameters and mixture frequencies."""

    w_sphericity: np.ndarray
    w_aspect_ratio: np.ndarray
    w_convexity: np.ndarray
    history: np.ndarray
    best_params: ParamsT
    best_frequencies: Optional[np.ndarray] = None
    effective_sample_size: Optional[float] = None
    stop_reason: Optional[dict] = None


def select_best_epoch(
    checkpoint_paths: Sequence[PathLike],
    optimal_distribution: np.ndarray,
    mesh_pipeline: MeshPipeline,
    *,
    n_samples: int = 24,
    work_dir: Optional[PathLike] = None,
    device: Optional[str] = None,
    batch_limit: int = 32,
    num_workers: Optional[int] = None,
    factor_weights: Optional[Sequence[float]] = None,
    seed: int = 0,
    threshold: float = 0.5,
    progress: bool = True,
    criterion: str = "mean",
    factor_scales: Optional[Sequence[float] | str] = None,
) -> OptimisationResult[Path]:
    """Screen generator checkpoints and return the best one and its cost history.

    Each checkpoint generates the same small, seeded batch. Its unoptimised
    shape-factor distribution is compared with the target using the weighted
    mean Wasserstein distance. Distances stay in each factor's own units
    unless *factor_scales* is given; note that ``criterion="minimax"`` on raw
    distances is dominated by whichever factor has the widest range, so pass
    ``factor_scales="std"`` (or explicit divisors) when using it. This is deliberately cheaper than running
    :class:`ShapeOptimiser` for every epoch; run the full generation and subset
    optimisation once, using the returned checkpoint.

    ``checkpoint_paths`` may be strided (for example, ``paths[::10]``) for an
    even faster coarse search. Screening files are created in a temporary
    directory and removed before this function returns.
    """
    checkpoints = [Path(path) for path in checkpoint_paths]
    if not checkpoints:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")

    if criterion not in ["mean", "minimax"]:
        raise ValueError("criterion must be 'mean' or 'minimax'")

    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if checkpoint.suffix.lower() not in (".pt", ".pth"):
            raise ValueError(f"Checkpoint must be a .pt or .pth file: {checkpoint}")
    if int(n_samples) != n_samples or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")

    target = _as_sample_rows(optimal_distribution).T
    if target.shape[0] != 3:
        raise ValueError("optimal_distribution must contain all three shape factors")
    weights = (
        np.ones(3)
        if factor_weights is None
        else np.asarray(factor_weights, dtype=float)
    )
    if weights.shape != (3,) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("factor_weights must contain three non-negative values")
    weights = weights / weights.sum()

    if factor_scales is None:
        scales = np.ones(3)
    elif isinstance(factor_scales, str):
        if factor_scales != "std":
            raise ValueError("factor_scales must be 'std', None, or three values")
        scales = target_factor_scales(target)
    else:
        scales = np.asarray(factor_scales, dtype=float)
        if (
            scales.shape != (3,)
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise ValueError("factor_scales must contain three positive values")

    root = Path(work_dir) if work_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    costs = np.zeros((3, len(checkpoints)), dtype=float)
    candidates = checkpoints if progress else checkpoints
    with TemporaryDirectory(prefix="gandem_epoch_screen_", dir=root) as temporary:
        for index, checkpoint in tqdm(
            enumerate(candidates),
            disable=not progress,
            desc="Screening epochs",
            unit="epoch",
        ):
            # Identical latent samples make checkpoint scores directly comparable.
            torch.manual_seed(seed)
            candidate_dir = Path(temporary) / str(index)
            pool = GeneratedParticlePool(n_samples, candidate_dir, mesh_pipeline)
            pool.generate(
                str(checkpoint),
                device=device,
                batch_limit=batch_limit,
                n_workers=num_workers,
                threshold=threshold,
                progress=False,
            )
            generated = pool.populate_shape_factors(
                num_workers=num_workers,
                rotations_per_stl=0,
                views_per_rotation=3,
                progress=False,
            )
            distances = np.asarray(
                [stats.wasserstein_distance(generated[i], target[i]) for i in range(3)]
            )

            costs[:, index] = distances

    scaled = costs / np.asarray(scales, dtype=float)[:, None]
    if criterion == "mean":
        history = np.dot(weights, scaled)
    else:
        history = np.max(scaled, axis=0)

    return OptimisationResult(
        w_sphericity=costs[0],
        w_aspect_ratio=costs[1],
        w_convexity=costs[2],
        history=history,
        best_params=checkpoints[int(np.argmin(history))],
    )


class ShapeOptimiser:
    """Select exactly ``n_select`` whole STLs to match a target distribution.

    Selection minimises either the weighted mean one-dimensional Wasserstein
    distance across sphericity, aspect ratio, and convexity, or their joint
    multivariate energy distance. A random-key CMA-ES maps each continuous
    candidate to the ``n_select`` highest-scoring STL IDs, while additional
    logits optimise their mixture frequencies. Every evaluation therefore
    contains exactly the requested number of whole meshes and a frequency
    vector that sums to one.

    Because the 2D objective is blind to 3D geometry, unconstrained mixture
    frequencies are the main route by which a selection overfits the 2D
    projections: the optimiser concentrates mass on a few particles to tune
    the 2D match, and the 3D shape-factor distribution then collapses onto
    that handful of meshes. ``logit_scale`` bounds this. It defaults to zero,
    which fixes the frequencies uniform and drops them from the search space,
    so the effective sample size always equals ``n_select``. Raise it to trade
    3D fidelity for a closer 2D match, checking
    :attr:`effective_sample_size` as you go; the maximum ratio between any two
    frequencies is ``exp(2 * logit_scale)``.

    Args:
        optimal_distribution: Target 2D observations, shaped ``(samples, 3)``
            or ``(3, samples)``.
        generations_pool: A :class:`GeneratedParticlePool` with populated
            shape factors, or an array shaped ``(3, observations)`` or
            ``(4, observations)`` whose fourth row holds STL IDs.
        n_select: Number of whole STLs to select.
        max_passes: Maximum CMA-ES generations.
        factor_weights: Relative contribution of each shape factor. These
            weight the marginal Wasserstein costs directly, or the squared
            coordinate differences in the energy-distance geometry. Defaults
            to equal weighting.
        population_size: CMA-ES population size. Defaults to the pycma value.
        sigma: Initial CMA-ES step size.
        covariance: ``"auto"``, ``"full"``, or ``"diagonal"``.
        polish: Refine the scalar result to a one-swap local optimum. This is
            a purely 2D greedy step; disable it if 3D fidelity matters more
            than the last increment of 2D fit.
        logit_scale: Bound on the mixture-frequency logits. Zero (the
            default) gives uniform frequencies.
        factor_scales: Divisor applied to each factor's Wasserstein distance
            before ``factor_weights`` is applied, or to each coordinate before
            computing energy distance. ``None`` (the default) leaves values in
            raw units. ``"std"`` divides by the target's per-factor standard
            deviation, and a sequence of three positive values supplies the
            divisors directly.
        initialisation: Starting point for CMA-ES. ``"stratified"`` (the
            default) begins from a subset spanning the pool, ``"greedy"``
            reproduces the earlier individual-cost ranking, and ``"neutral"``
            begins from an unbiased mean.
        distance_metric: ``"wasserstein"`` (the default) compares the three
            marginal distributions. ``"energy"`` compares the joint
            three-dimensional distribution and therefore detects dependencies
            between shape factors.

    Note:
        A Wasserstein distance is expressed in the units of its own shape
        factor, and the three factors do not span comparable ranges. With raw
        distances and equal ``factor_weights``, aspect ratio typically counts
        several times more than convexity, so the weights do not mean what
        they appear to. ``factor_scales="std"`` removes that bias -- but it is
        not the default, because on representative pools it trades the bias
        for a worse problem: convexity is the narrowest factor and carries the
        least signal, so promoting it to equal footing makes the optimiser
        drive convexity below the distance a random subset of the same size
        achieves while letting aspect ratio become about twice as bad as
        random. Change the scales only alongside a random-subset comparison
        that shows the result actually improved.

    Note:
        The starting point matters more than the objective here. Measured on
        synthetic pools whose 3D distribution already matches ground truth,
        the ``"greedy"`` initialisation produces a selection worse in 3D than
        94% of random subsets of the same size before any search happens,
        because ranking particles by how well each one alone reproduces the
        whole target systematically demotes the elongated and near-spherical
        meshes that fill the tails. The search needs roughly twenty
        generations to recover from that. ``"stratified"`` starts at about the
        24th percentile instead and ends near the 5th, and it degrades far
        more gracefully when a run is cut short.

        :attr:`best_cost` and ``result.history`` contain the configured
        objective. :meth:`factor_distances` and the per-factor histories on
        :class:`OptimisationResult` remain Wasserstein diagnostics in the
        native units of each factor, including when ``distance_metric`` is
        ``"energy"``.
    """

    def __init__(
        self,
        optimal_distribution: np.ndarray,
        generations_pool: GeneratedParticlePool | np.ndarray,
        n_select: int,
        max_passes: int = 100,
        factor_weights: Optional[Sequence[float]] = None,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        covariance: str = "auto",
        polish: bool = True,
        logit_scale: float = 0.0,
        factor_scales: Optional[Sequence[float] | str] = None,
        initialisation: str = "stratified",
        distance_metric: str = "wasserstein",
    ):
        target = _as_sample_rows(optimal_distribution).T
        if target.shape[0] != 3:
            raise ValueError(
                "optimal_distribution must contain all three shape factors"
            )
        self.optimal_distribution = target

        if factor_scales is None:
            scales = np.ones(3)
        elif isinstance(factor_scales, str):
            if factor_scales != "std":
                raise ValueError("factor_scales must be 'std', None, or three values")
            scales = target_factor_scales(target)
        else:
            scales = np.asarray(factor_scales, dtype=float)
            if (
                scales.shape != (3,)
                or not np.all(np.isfinite(scales))
                or np.any(scales <= 0)
            ):
                raise ValueError("factor_scales must contain three positive values")
        self.factor_scales = scales

        self.generations_pool = (
            generations_pool
            if isinstance(generations_pool, GeneratedParticlePool)
            else None
        )
        population = (
            generations_pool.shape_factors
            if isinstance(generations_pool, GeneratedParticlePool)
            else np.asarray(generations_pool, dtype=float)
        )
        if population is None:
            raise ValueError(
                "call populate_shape_factors() before constructing ShapeOptimiser"
            )
        if population.ndim != 2 or population.shape[0] not in (3, 4):
            raise ValueError(
                "generations_pool must have shape (3, observations) or (4, observations)"
            )
        if population.shape[0] == 3:
            population = np.vstack((population, np.arange(population.shape[1])))
        if not np.all(np.isfinite(population)):
            raise ValueError("generations_pool contains non-finite shape factors")
        self.total_population = population
        self.particle_ids = np.unique(population[3].astype(int))

        if int(n_select) != n_select or not 1 <= n_select <= len(self.particle_ids):
            raise ValueError(f"n_select must be between 1 and {len(self.particle_ids)}")
        if max_passes < 0:
            raise ValueError("max_passes must be non-negative")
        self.n_select = int(n_select)
        self.max_passes = int(max_passes)

        if population_size is not None and (
            int(population_size) != population_size or population_size < 2
        ):
            raise ValueError("population_size must be an integer of at least 2")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if covariance not in ("auto", "full", "diagonal"):
            raise ValueError("covariance must be 'auto', 'full', or 'diagonal'")
        if not np.isfinite(logit_scale) or logit_scale < 0:
            raise ValueError("logit_scale must be non-negative and finite")
        if initialisation not in INITIALISATIONS:
            raise ValueError(
                "initialisation must be " + ", ".join(map(repr, INITIALISATIONS))
            )
        if distance_metric not in DISTANCE_METRICS:
            raise ValueError(
                "distance_metric must be " + ", ".join(map(repr, DISTANCE_METRICS))
            )
        self.population_size = (
            int(population_size) if population_size is not None else None
        )
        self.sigma = float(sigma)
        self.covariance = covariance
        self.polish = bool(polish)
        self.logit_scale = float(logit_scale)
        self._frequency_dimensions = 0 if self.logit_scale == 0 else self.n_select
        self.initialisation = initialisation
        self.distance_metric = distance_metric

        weights = (
            np.ones(3)
            if factor_weights is None
            else np.asarray(factor_weights, dtype=float)
        )
        if weights.shape != (3,) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("factor_weights must contain three non-negative values")
        self.factor_weights = weights / weights.sum()

        self._energy_scale = np.sqrt(self.factor_weights) / self.factor_scales
        self._energy_target = (
            self.optimal_distribution.T * self._energy_scale
            if self.distance_metric == "energy"
            else None
        )
        self._energy_target_within = (
            float(2 * pdist(self._energy_target).sum() / len(self._energy_target) ** 2)
            if self._energy_target is not None
            else None
        )

        self.best_idxs: Optional[np.ndarray] = None
        self.best_frequencies: Optional[np.ndarray] = None
        self.best_cost = float("inf")
        self.stop_reason: Optional[dict] = None

    @property
    def effective_sample_size(self) -> Optional[float]:
        """Kish effective sample size ``1 / sum(w ** 2)`` of the best mixture.

        Returns ``None`` before :meth:`optimise` has run. A value well below
        :attr:`n_select` means the mixture has collapsed onto a few particles,
        so the selection represents far fewer distinct shapes than it appears
        to. Raise :attr:`logit_scale` only as far as this diagnostic allows.
        """
        if self.best_frequencies is None:
            return None
        weights = np.asarray(self.best_frequencies, dtype=float)
        total = float(np.sum(weights**2))
        return float(1.0 / total) if total > 0 else None

    def _observations(self, particle_ids: Sequence[int]) -> np.ndarray:
        mask = np.isin(self.total_population[3].astype(int), particle_ids)
        return self.total_population[:3, mask]

    def _observation_weights(
        self,
        particle_ids: Sequence[int],
        particle_frequencies: Sequence[float],
    ) -> np.ndarray:
        """Split each particle's mixture frequency equally across its views."""
        ids = np.asarray(particle_ids, dtype=int)
        observation_ids = self.total_population[3].astype(int)
        frequencies = np.asarray(particle_frequencies, dtype=float)
        counts = np.asarray(
            [np.count_nonzero(observation_ids == particle_id) for particle_id in ids]
        )
        return np.repeat(frequencies / counts, counts)

    def _factor_distance_values(
        self,
        particle_ids: Sequence[int],
        particle_frequencies: Sequence[float],
    ) -> np.ndarray:
        ids = np.asarray(particle_ids, dtype=int)
        observation_ids = self.total_population[3].astype(int)
        columns = np.concatenate(
            [np.flatnonzero(observation_ids == particle_id) for particle_id in ids]
        )
        selected = self.total_population[:3, columns]
        weights = self._observation_weights(ids, particle_frequencies)
        return np.asarray(
            [
                stats.wasserstein_distance(
                    selected[i], self.optimal_distribution[i], u_weights=weights
                )
                for i in range(3)
            ]
        )

    def _energy_distance(
        self,
        particle_ids: Sequence[int],
        particle_frequencies: Sequence[float],
    ) -> float:
        """Return weighted energy distance between joint shape-factor samples."""
        assert self._energy_target is not None
        assert self._energy_target_within is not None
        ids = np.asarray(particle_ids, dtype=int)
        observation_ids = self.total_population[3].astype(int)
        columns = np.concatenate(
            [np.flatnonzero(observation_ids == particle_id) for particle_id in ids]
        )
        selected = self.total_population[:3, columns].T * self._energy_scale
        weights = self._observation_weights(ids, particle_frequencies)
        weights = weights / weights.sum()

        cross = float(
            np.sum(cdist(selected, self._energy_target) * weights[:, None])
            / len(self._energy_target)
        )
        within = float(np.sum(cdist(selected, selected) * np.outer(weights, weights)))
        squared = 2 * cross - within - self._energy_target_within
        return float(np.sqrt(max(0.0, squared)))

    def factor_distances(
        self,
        particle_ids: Optional[Sequence[int]] = None,
        particle_frequencies: Optional[Sequence[float]] = None,
    ) -> dict[str, float]:
        """Return an interpretable Wasserstein distance for each shape factor.

        Distances are in the native units of each factor, unscaled by
        :attr:`factor_scales`, so they can be compared against the factor
        values themselves. The Wasserstein objective minimises their scaled,
        weighted mean; the energy objective reports them only as diagnostics.
        Use them for comparing a selection against the whole pool.

        When *particle_ids* is omitted, the best selection from :meth:`optimise`
        is used.
        """
        if particle_ids is None:
            if self.best_idxs is None:
                raise RuntimeError("call optimise() before requesting diagnostics")
            ids = self.best_idxs
            frequency_values = self.best_frequencies
        else:
            ids = np.asarray(particle_ids, dtype=int)
            frequency_values = particle_frequencies
        unknown = np.setdiff1d(ids, self.particle_ids)
        if unknown.size:
            raise ValueError(f"unknown particle ID: {unknown[0]}")
        frequencies = (
            np.full(len(ids), 1 / len(ids))
            if frequency_values is None
            else np.asarray(frequency_values, dtype=float)
        )
        if (
            frequencies.shape != (len(ids),)
            or np.any(frequencies < 0)
            or not np.all(np.isfinite(frequencies))
            or frequencies.sum() <= 0
        ):
            raise ValueError("particle_frequencies must be finite non-negative values")
        frequencies = frequencies / frequencies.sum()
        return dict(
            zip(
                SHAPE_FACTORS,
                self._factor_distance_values(ids, frequencies).tolist(),
            )
        )

    def _scaled(self, distances: np.ndarray) -> np.ndarray:
        """Divide raw factor distances by :attr:`factor_scales`.

        Only the optimiser sees scaled distances. Everything reported back --
        :meth:`factor_distances` and the per-factor histories -- stays in the
        native units of each shape factor.
        """
        return np.asarray(distances, dtype=float) / self.factor_scales

    def _w_cost(
        self,
        particle_ids: Sequence[int],
        particle_frequencies: Sequence[float],
    ) -> float:
        if self.distance_metric == "energy":
            return self._energy_distance(particle_ids, particle_frequencies)
        distances = self._factor_distance_values(particle_ids, particle_frequencies)
        return float(np.dot(self.factor_weights, self._scaled(distances)))

    def _decode(self, values: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
        """Decode selection keys and bounded frequency logits.

        :attr:`logit_scale` bounds how far the mixture may depart from uniform.
        Raw CMA-ES coordinates are squashed with ``tanh`` into
        ``[-logit_scale, logit_scale]``, so the largest and smallest frequency
        differ by a factor of at most ``exp(2 * logit_scale)``. A scale of zero
        removes the frequency coordinates from the search space entirely and
        returns uniform frequencies, keeping the effective sample size equal to
        ``n_select``.
        """
        keys = values[: len(self.particle_ids)]
        positions = np.argpartition(keys, -self.n_select)[-self.n_select :]
        particle_ids = tuple(sorted(self.particle_ids[positions].tolist()))
        if not self._frequency_dimensions:
            return particle_ids, np.full(self.n_select, 1.0 / self.n_select)
        logits = self.logit_scale * np.tanh(values[len(self.particle_ids) :])
        exponentials = np.exp(logits - logits.max())
        return particle_ids, exponentials / exponentials.sum()

    def _cma_options(self, seed: Optional[int], diagonal: bool) -> dict:
        options = {
            "CMA_diagonal": diagonal,
            "tolflatfitness": FLAT_FITNESS_TOLERANCE,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        if seed is not None:
            options["seed"] = int(seed) or 1
        if self.population_size is not None:
            options["popsize"] = self.population_size
        return options

    def _particle_descriptors(self) -> np.ndarray:
        """Mean shape factors of each particle's own projections, ``(P, 3)``."""
        observation_ids = self.total_population[3].astype(int)
        return np.asarray(
            [
                self.total_population[:3, observation_ids == particle_id].mean(axis=1)
                for particle_id in self.particle_ids
            ]
        )

    def _stratified_positions(self, seed: Optional[int]) -> np.ndarray:
        """Choose ``n_select`` positions spanning the pool's dominant axis.

        Particles are described by the mean of their own projections, projected
        onto the first principal component of those descriptors, then split by
        rank into ``n_select`` equally sized strata with one particle drawn
        from each. The result covers the pool's range in proportion to its
        density, so the extreme particles that populate the tails of the 3D
        distribution are represented from the first generation.

        Args:
            seed: Seed for the within-stratum draw. Different seeds give
                different spanning subsets, which makes multi-restart runs
                genuinely independent.

        Returns:
            Positions into :attr:`particle_ids`, of length ``n_select``.
        """
        descriptors = self._particle_descriptors()
        if len(descriptors) == 1:
            return np.zeros(1, dtype=int)

        spread = np.std(descriptors, axis=0)
        centred = (descriptors - descriptors.mean(axis=0)) / np.where(
            spread > 0, spread, 1.0
        )
        if np.allclose(centred, 0.0):
            key = np.arange(len(centred), dtype=float)
        else:
            components = np.linalg.svd(centred, full_matrices=False)[2]
            key = centred @ components[0]

        rng = np.random.default_rng(seed)
        order = np.argsort(key, kind="stable")
        picks = [
            int(rng.choice(stratum))
            for stratum in np.array_split(order, self.n_select)
            if stratum.size
        ]
        remaining = np.setdiff1d(order, picks)
        rng.shuffle(remaining)
        picks.extend(int(item) for item in remaining[: self.n_select - len(picks)])
        return np.asarray(picks[: self.n_select], dtype=int)

    def _initial_mean(
        self,
        seed: Optional[int] = None,
        *,
        progress: bool = False,
    ) -> np.ndarray:
        """Build the CMA-ES starting mean for the configured initialisation.

        ``"stratified"`` (the default) starts from a subset spanning the pool.
        ``"greedy"`` ranks particles by how well each one alone matches the
        whole target, which measurably depletes the tails of the selected 3D
        distribution -- it is retained only for reproducing earlier results.
        ``"neutral"`` starts from an unbiased mean.
        """
        if self.initialisation == "neutral":
            keys = np.zeros(len(self.particle_ids))
        elif self.initialisation == "greedy":
            particle_ids = (
                tqdm(
                    self.particle_ids,
                    desc="Initialising selection",
                    unit="STL",
                )
                if progress
                else self.particle_ids
            )
            costs = np.asarray([self._w_cost([item], [1.0]) for item in particle_ids])
            order = np.argsort(np.argsort(costs, kind="stable"), kind="stable")
            if len(order) == 1:
                keys = np.zeros(1)
            else:
                keys = 1 - 2 * order / (len(order) - 1)
        else:
            # One step size of separation seeds the subset without pinning it:
            # a wider gap measurably slows CMA-ES's escape from the start.
            keys = np.full(len(self.particle_ids), -self.sigma)
            keys[self._stratified_positions(seed)] = self.sigma
        return np.concatenate((keys, np.zeros(self._frequency_dimensions)))

    def _polish(
        self,
        selected: tuple[int, ...],
        frequencies: np.ndarray,
        cost: float,
    ) -> tuple[tuple[int, ...], float]:
        """Refine a CMA-ES subset to a deterministic one-swap local optimum."""
        current = list(selected)
        while True:
            best = (cost, tuple(current))
            available = sorted(set(self.particle_ids) - set(current))
            for position in range(self.n_select):
                for candidate in available:
                    trial = current.copy()
                    trial[position] = candidate
                    trial_ids = tuple(sorted(trial))
                    trial_cost = self._w_cost(trial_ids, frequencies)
                    if (trial_cost, trial_ids) < best:
                        best = trial_cost, trial_ids
            if best[0] >= cost:
                return tuple(current), cost
            cost, current = best[0], list(best[1])

    def _scalar_result(
        self,
        history: list[float],
        factor_history: list[np.ndarray],
    ) -> OptimisationResult[np.ndarray]:
        """Package scalar CMA-ES diagnostics and the selected STL IDs."""
        assert self.best_idxs is not None
        assert self.best_frequencies is not None
        factors = np.asarray(factor_history)
        return OptimisationResult(
            w_sphericity=factors[:, 0],
            w_aspect_ratio=factors[:, 1],
            w_convexity=factors[:, 2],
            history=np.asarray(history),
            best_params=self.best_idxs.copy(),
            best_frequencies=self.best_frequencies.copy(),
            effective_sample_size=self.effective_sample_size,
            stop_reason=dict(self.stop_reason) if self.stop_reason else None,
        )

    def optimise(
        self,
        seed: Optional[int] = None,
        *,
        progress: bool = False,
    ) -> OptimisationResult[np.ndarray]:
        """Run CMA-ES using the configured distance metric.

        Full covariance adaptation is used for pools up to 64 STLs. Larger
        pools automatically use separable (diagonal) CMA-ES to avoid quadratic
        memory and cubic eigendecompositions. The selected values in
        :attr:`best_idxs` are STL IDs, not observation column indices.

        Returns best-so-far factor diagnostics and objective costs by
        generation in an :class:`OptimisationResult`. The per-factor histories
        are Wasserstein distances in native units. ``history`` contains the
        configured Wasserstein or energy-distance objective. The optimised
        mixture proportions are returned in ``result.best_frequencies`` and
        :attr:`best_frequencies`, aligned with ``result.best_params``.

        Args:
            seed: Random seed for reproducible selection.
            progress: Display initialization and generation progress when ``True``.

        Note:
            :attr:`stop_reason` records the pycma termination criterion that
            ended the run, or stays ``None`` when ``max_passes`` was reached.
            Check it before concluding that a result converged: the random-key
            encoding makes the cost a step function, so pycma's flat-fitness
            criterion can fire long before the search is finished. It is
            configured with :data:`FLAT_FITNESS_TOLERANCE` generations of
            tolerance rather than pycma's default of one.
        """
        with _isolated_global_random_state():
            return self._optimise_scalar(seed, progress)

    def _optimise_scalar(
        self,
        seed: Optional[int],
        progress: bool,
    ) -> OptimisationResult[np.ndarray]:
        """Run scalar CMA-ES over the random-key encoding."""
        self.stop_reason = None
        mean = self._initial_mean(seed, progress=progress)
        dimension = len(mean)
        best_ids, best_frequencies = self._decode(mean)
        best_distances = self._factor_distance_values(best_ids, best_frequencies)
        best_cost = self._w_cost(best_ids, best_frequencies)
        history = [best_cost]
        factor_history = [best_distances]
        if self.max_passes == 0 or len(self.particle_ids) == 1:
            self.best_idxs = np.asarray(best_ids)
            self.best_frequencies = best_frequencies
            self.best_cost = best_cost
            return self._scalar_result(history, factor_history)

        diagonal = self.covariance == "diagonal" or (
            self.covariance == "auto" and dimension > 64
        )
        optimiser = CMAEvolutionStrategy(
            mean,
            self.sigma,
            self._cma_options(seed, diagonal),
        )
        generations = (
            trange(self.max_passes, desc="Optimising selection", unit="generation")
            if progress
            else range(self.max_passes)
        )
        for _ in generations:
            candidates = optimiser.ask()
            costs = []
            generation_best = (
                float("inf"),
                best_ids,
                best_frequencies,
            )
            for values in candidates:
                particle_ids, frequencies = self._decode(values)
                cost = self._w_cost(particle_ids, frequencies)
                costs.append(cost)
                if (cost, particle_ids) < generation_best[:2]:
                    generation_best = cost, particle_ids, frequencies
            optimiser.tell(candidates, costs)

            generation_cost, generation_ids, generation_frequencies = generation_best
            if (generation_cost, generation_ids) < (best_cost, best_ids):
                best_ids, best_cost = generation_ids, generation_cost
                best_frequencies = generation_frequencies
                best_distances = self._factor_distance_values(
                    best_ids, best_frequencies
                )
            history.append(best_cost)
            factor_history.append(best_distances)
            reason = optimiser.stop()
            if reason:
                self.stop_reason = dict(reason)
                break

        if self.polish:
            best_ids, best_cost = self._polish(best_ids, best_frequencies, best_cost)
            if best_cost < history[-1]:
                history.append(best_cost)
                best_distances = self._factor_distance_values(
                    best_ids, best_frequencies
                )
                factor_history.append(best_distances)
        self.best_idxs = np.asarray(best_ids, dtype=int)
        self.best_frequencies = best_frequencies
        self.best_cost = best_cost
        return self._scalar_result(history, factor_history)

    @property
    def selected_files(self) -> list[Path]:
        """Paths of the selected STL files."""
        if self.best_idxs is None:
            raise RuntimeError("call optimise() before requesting selected_files")
        if self.generations_pool is None:
            raise ValueError(
                "STL paths are available only when using a GeneratedParticlePool"
            )
        return [self.generations_pool.stl_files[index] for index in self.best_idxs]

    def copy_selected(self, output_dir: PathLike) -> list[Path]:
        """Copy the selected STL files to *output_dir* and return new paths."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        return [
            Path(shutil.copy2(path, destination / path.name))
            for path in self.selected_files
        ]

    def ks_validate(self, alpha: float = 0.05) -> KSResults:
        """Run a two-sample KS test for each selected shape-factor frequency."""
        if self.best_idxs is None:
            raise RuntimeError("call optimise() before validation")
        selected = self._observations(self.best_idxs)
        return KSResults(
            *(
                stats.ks_2samp(selected[i], self.optimal_distribution[i]).pvalue > alpha
                for i in range(3)
            )
        )
