"""Mesh and voxel-grid processing operations."""

from abc import ABC, abstractmethod
import os
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence
from warnings import warn
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
import trimesh as tm
from .shape_factors import _shape_factors_3d
import pymeshfix
import pyvista as pv

from ..ganular.artifacts import save_voxel_stl
from ..ganular.quality import mesh_genus


def _require_mesh_repair_dependencies() -> None:
    if pymeshfix is None or pv is None:
        raise ImportError(
            "Mesh repair requires the optional 'pymeshfix' and 'pyvista' packages"
        )


def _mesh_from_probability_grid(
    probability_grid: np.ndarray, threshold: float, *, autoclean: bool
) -> tm.Trimesh:
    """Extract one isosurface and optionally repair it with PyMeshFix."""
    vertices, faces, _, _ = measure.marching_cubes(probability_grid, level=threshold)
    if not autoclean:
        return tm.Trimesh(vertices=vertices, faces=faces)

    _require_mesh_repair_dependencies()
    meshfix = pymeshfix.MeshFix(vertices, faces)
    meshfix.repair()
    meshfix.clean()
    return pv.to_trimesh(meshfix.mesh, triangulate=True)


class MeshProcess(ABC):
    """Abstract base class for a single mesh processing operation.

    Subclasses must implement :meth:`apply` to transform a
    ``trimesh.Trimesh`` object in place or return a new one.
    """

    def __repr__(self):
        return "MeshProcess"

    @abstractmethod
    def apply(self, mesh: tm.Trimesh) -> tm.Trimesh:
        """Apply the processing step to a mesh.

        Args:
            mesh: Input triangle mesh.

        Returns:
            Processed triangle mesh.
        """
        pass


class GridProcess(ABC):
    """Abstract base class for converting a grid to a mesh.

    Subclasses must implement :meth:`apply` to extract a mesh from a
    3D numpy array.
    """

    def __repr__(self):
        return "GridProcess"

    @abstractmethod
    def apply(self, grid: np.ndarray, /) -> tm.Trimesh:
        """Extract a mesh from a 3D grid.

        Args:
            grid: Input 3D numpy array.

        Returns:
            Extracted triangle mesh.
        """
        pass


class MeshPipeline:
    """Sequential pipeline that applies a chain of mesh processing steps.

    Each :class:`MeshProcess` is executed in order; the output of one step
    becomes the input of the next.

    Args:
        mesh_processes: Ordered list of :class:`MeshProcess` instances to
            apply.
    """

    def __init__(self, mesh_processes: Sequence[GridProcess | MeshProcess]):
        processes = list(mesh_processes)
        for i, process in enumerate(processes):
            if not isinstance(process, (GridProcess, MeshProcess)):
                raise TypeError(
                    f"Invalid process type at index {i}: {type(process).__name__}. "
                    "Processes must be instances of MeshProcess or GridProcess."
                )
            if i == 0 and not isinstance(process, GridProcess):
                raise ValueError(
                    "First process must be a GridProcess to convert the probability "
                    "grid to a mesh,"
                    f" but got {type(process).__name__}."
                )
            if i > 0 and isinstance(process, GridProcess):
                raise ValueError(
                    f"GridProcess found at index {i} after the first process. "
                    "All subsequent processes must be MeshProcess instances."
                )

        self.mesh_processes = processes

    def execute(
        self,
        grid: np.ndarray | str | PathLike,
        save_path: Optional[str | PathLike] = None,
    ) -> Optional[tm.Trimesh]:
        """Convert a probability grid to a mesh, optionally exporting it.

        Args:
            grid: A 3D probability grid or path to a ``.npy`` file. A file
                input is exported beside the source by default.
            save_path: Optional STL output path. Array inputs are not exported
                when this is omitted.

        Returns:
            The processed mesh, or ``None`` when processing yields no faces.

        Raises:
            FileNotFoundError: If a file path is given and the file does
                not exist.
        """
        if isinstance(grid, (str, os.PathLike)):
            grid_path = Path(grid)
            if not grid_path.is_file():
                raise FileNotFoundError(f"Grid file '{grid_path}' does not exist.")
            save_path = save_path or grid_path.with_suffix(".stl")
            grid = np.load(grid_path)

        if not self.mesh_processes:
            if save_path is None:
                raise ValueError("save_path is required for an empty MeshPipeline")
            warn("No mesh processes provided; saving the voxel grid directly.")
            save_voxel_stl(grid, save_path)
            return None

        first, *remaining = self.mesh_processes
        assert isinstance(first, GridProcess)
        msh = first.apply(np.asarray(grid).copy())
        if getattr(msh, "faces", None) is None or len(msh.faces) == 0:
            return None
        for process in remaining:
            assert isinstance(process, MeshProcess)
            msh = process.apply(msh)
            if getattr(msh, "faces", None) is None or len(msh.faces) == 0:
                return None

        if save_path is not None:
            msh.export(save_path)
        return msh


class MeshRepair(MeshProcess):
    """
    Repair a mesh to make it watertight and fix common issues like inverted faces and holes.
    Wraps the PyMeshFix library to perform the repairs.

    Args:
    max_iters: Maximum number of iterations for the cleaning process.
    inner_loops: Number of inner loops for the cleaning process.


    """

    def __init__(self, max_iters: int = 10, inner_loops: int = 3):
        self.max_iters = max_iters
        self.inner_loops = inner_loops

    def apply(self, mesh: tm.Trimesh) -> tm.Trimesh:
        """Fix normals, inversions, and fill holes in the mesh.

        Args:
            mesh: Input triangle mesh.

        Returns:
            Repaired triangle mesh (copy of the original).
        """
        _require_mesh_repair_dependencies()
        vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
        mfix = pymeshfix.MeshFix(vertices, faces)
        mfix.repair()
        mfix.clean(self.max_iters, self.inner_loops)

        polydata_mesh = mfix.mesh
        repaired_mesh = pv.to_trimesh(polydata_mesh, triangulate=True)

        # Normalize the VTK-to-Trimesh conversion before repairing any
        # boundaries left by MeshFix's degeneracy/intersection cleanup.
        repaired_mesh.process(validate=True)
        repaired_mesh.merge_vertices()
        repaired_mesh.remove_unreferenced_vertices()
        tm.repair.fill_holes(repaired_mesh)
        tm.repair.fix_normals(repaired_mesh)
        tm.repair.fix_inversion(repaired_mesh)

        return repaired_mesh


class ReduceTriangles(MeshProcess):
    """Simplify a mesh by reducing its triangle count via quadratic decimation.

    Args:
        target_reduction_frac: Fraction of faces to remove (0--1).
        target_face_count: Absolute target number of faces.
        aggression: Controls the trade-off between speed and quality during
            decimation.
    """

    def __init__(
        self,
        target_reduction_frac: Optional[float] = None,
        target_face_count: Optional[int] = None,
        aggression: Optional[int] = None,
    ):
        self.percent = target_reduction_frac
        self.face_count = target_face_count
        self.aggression = aggression

    def apply(self, mesh: tm.Trimesh) -> tm.Trimesh:
        """Reduce the triangle count of the mesh.

        Args:
            mesh: Input triangle mesh.

        Returns:
            Simplified triangle mesh.
        """
        return mesh.simplify_quadric_decimation(
            self.percent, self.face_count, self.aggression
        )


class SmoothMesh(MeshProcess):
    """Smooth a mesh surface using Laplacian smoothing.

    Args:
        iterations: Number of smoothing passes.
        lambda_: Smoothing factor. Higher values produce more aggressive
            smoothing.
    """

    def __init__(
        self, iterations: Optional[int] = None, lambda_: Optional[float] = None
    ):
        self.iterations = iterations
        self.lamb = lambda_

    def apply(self, mesh: tm.Trimesh) -> tm.Trimesh:
        """Apply Laplacian smoothing to the mesh.

        Args:
            mesh: Input triangle mesh.

        Returns:
            Smoothed triangle mesh.
        """
        return tm.smoothing.filter_mut_dif_laplacian(
            mesh, iterations=self.iterations, lamb=self.lamb
        )


class Otsu3DThreshold(GridProcess):
    """Extract a mesh using an Otsu threshold.

    Args:
        n_bins: Histogram resolution. 256 is standard.
        autoclean: Repair the extracted mesh with the optional PyMeshFix and
            PyVista dependencies.
    """

    def __init__(self, n_bins: int = 256, autoclean: bool = False):
        self.n_bins = n_bins
        self.threshold: float = float("nan")
        self.best_thresh: float = float("nan")
        self.autoclean = autoclean

    def _find_otsu_threshold(
        self, probability_grid: np.ndarray, n_bins: int = 256
    ) -> float:
        hist, bin_edges = np.histogram(
            probability_grid.ravel(), bins=n_bins, range=(0, 1)
        )

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight0 = np.cumsum(hist)
        weight1 = np.cumsum(hist[::-1])[::-1]

        cumulative_intensity_sum = np.cumsum(hist * bin_centers)
        total_intensity_sum = cumulative_intensity_sum[-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            mean0 = cumulative_intensity_sum / weight0
            mean1 = (total_intensity_sum - cumulative_intensity_sum) / weight1

        # inter-class variance
        variance = weight0 * weight1 * (mean0 - mean1) ** 2

        # optimal index
        idx = np.nanargmax(variance)

        self.threshold = self.best_thresh = float(bin_centers[idx])
        return self.threshold

    def apply(self, probability_grid: np.ndarray) -> tm.Trimesh:
        """Extract and optionally repair the Otsu isosurface."""

        threshold = self._find_otsu_threshold(probability_grid, self.n_bins)
        return _mesh_from_probability_grid(
            probability_grid, threshold, autoclean=self.autoclean
        )


class TopologyAware3DSegmenter(GridProcess):
    """Convert a 3D probability volume into a clean particle mesh.

    The default configuration targets the common EvoPop case: one particle
    in a ``64 x 64 x 64`` probability volume. Segmentation uses adaptive
    dual-threshold hysteresis, so uncertain boundary voxels are retained only
    when connected to a high-confidence particle core.

    Args:
        spacing: Physical voxel spacing in array-axis order.
        min_particle_volume: Smallest retained component, in physical units
            cubed. With unit spacing this is a voxel count.
        multiple_particles: Retain and label all supported particles instead
            of selecting the most probable component.
        thresholds: Optional explicit ``(low, high)`` hysteresis thresholds.
            By default they are estimated from the volume using Otsu's value
            as the centre of an uncertainty band.
        smoothing_sigma: Gaussian smoothing scale in physical units. Set to
            zero to disable probability smoothing.
        hysteresis_margin: Half-width around the automatically selected
            threshold.
        bridge_radius: Maximum physical radius used to reconnect short,
            probability-supported gaps. Zero disables bridging.
        fill_holes_up_to: Maximum physical volume of an enclosed cavity to
            fill. Zero preserves every cavity.
        split_touching: Apply distance-transform watershed splitting. When
            omitted, it follows ``multiple_particles``.
        watershed_h: Minimum physical prominence of watershed markers.
        surface_sigma: Signed-distance smoothing scale used only for mesh
            extraction. Topology-changing smoothing is automatically rejected.
        preserve_topology: Reject surface smoothing when component count,
            cavity count, or genus changes.
        dem_safe: Require a single genus-zero solid suitable for DEM. Narrow
            tunnels and cavities are filled before meshing; unsafe particles
            are rejected.
        max_tunnel_radius: Largest physical closing radius used for tunnel
            repair.
        max_repair_fraction: Maximum changed-voxel fraction accepted during
            DEM repair.
        autoclean: Run the existing PyMeshFix cleanup after mesh extraction.

    Notes:
        :meth:`segment` exposes the binary or labeled voxel morphology.
        :meth:`apply` returns a :class:`trimesh.Trimesh` so the class can be
        used directly as the first operation in :class:`MeshPipeline`.
    """

    def __init__(
        self,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_particle_volume: float = 8.0,
        multiple_particles: bool = False,
        *,
        thresholds: Optional[tuple[float, float]] = None,
        smoothing_sigma: float = 0.75,
        hysteresis_margin: float = 0.10,
        bridge_radius: float = 0.0,
        fill_holes_up_to: float = 8.0,
        split_touching: Optional[bool] = None,
        watershed_h: float = 2.0,
        surface_sigma: float = 0.5,
        preserve_topology: bool = True,
        dem_safe: bool = False,
        max_tunnel_radius: float = 4.0,
        max_repair_fraction: float = 0.20,
        autoclean: bool = False,
    ):
        self.spacing = np.asarray(spacing, dtype=float)
        if self.spacing.shape != (3,) or not np.all(self.spacing > 0):
            raise ValueError("spacing must contain three positive values")
        if min_particle_volume < 0:
            raise ValueError("min_particle_volume must be non-negative")
        if thresholds is not None and not (
            len(thresholds) == 2 and 0 <= thresholds[0] < thresholds[1] <= 1
        ):
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        for name, value in (
            ("smoothing_sigma", smoothing_sigma),
            ("hysteresis_margin", hysteresis_margin),
            ("bridge_radius", bridge_radius),
            ("fill_holes_up_to", fill_holes_up_to),
            ("watershed_h", watershed_h),
            ("surface_sigma", surface_sigma),
            ("max_tunnel_radius", max_tunnel_radius),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= max_repair_fraction <= 1:
            raise ValueError("max_repair_fraction must lie in [0, 1]")
        if dem_safe and multiple_particles:
            raise ValueError("dem_safe requires multiple_particles=False")

        self.min_particle_volume = float(min_particle_volume)
        self.multiple_particles = bool(multiple_particles)
        self.thresholds = thresholds
        self.smoothing_sigma = float(smoothing_sigma)
        self.hysteresis_margin = float(hysteresis_margin)
        self.bridge_radius = float(bridge_radius)
        self.fill_holes_up_to = float(fill_holes_up_to)
        self.split_touching = (
            self.multiple_particles if split_touching is None else bool(split_touching)
        )
        self.watershed_h = float(watershed_h)
        self.surface_sigma = float(surface_sigma)
        self.preserve_topology = bool(preserve_topology)
        self.dem_safe = bool(dem_safe)
        self.max_tunnel_radius = float(max_tunnel_radius)
        self.max_repair_fraction = float(max_repair_fraction)
        self.autoclean = bool(autoclean)

        self.last_thresholds: tuple[float, float] = (float("nan"), float("nan"))
        self.last_labels: Optional[np.ndarray] = None
        self.last_report: dict[str, object] = {}

    @property
    def _voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    @staticmethod
    def _empty_labels(shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.int32)

    def _prepare(self, probability_grid: np.ndarray) -> np.ndarray:
        probability = np.squeeze(np.asarray(probability_grid, dtype=np.float32))
        if probability.ndim != 3:
            raise ValueError(
                f"Expected a 3D probability volume, got shape {probability.shape}"
            )
        if probability.size == 0 or not np.isfinite(probability).all():
            raise ValueError("probability_grid must be non-empty and finite")
        if probability.min() < -1e-6 or probability.max() > 1.0 + 1e-6:
            raise ValueError("probability_grid values must lie in [0, 1]")
        return np.clip(probability, 0.0, 1.0)

    def _select_thresholds(self, probability: np.ndarray) -> tuple[float, float]:
        if self.thresholds is not None:
            return float(self.thresholds[0]), float(self.thresholds[1])

        midpoint = float(filters.threshold_otsu(probability))
        low = float(np.clip(midpoint - self.hysteresis_margin, 0.05, 0.85))
        high = float(np.clip(midpoint + self.hysteresis_margin, low + 0.02, 0.95))
        return low, high

    def _physical_ball(self, radius: float) -> np.ndarray:
        radii = np.ceil(radius / self.spacing).astype(int)
        shape = tuple(2 * radii + 1)
        coordinates = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
        distance_squared = sum(
            (axis * spacing / radius) ** 2
            for axis, spacing in zip(coordinates, self.spacing)
        )
        return np.asarray(distance_squared <= 1.0).reshape(shape)

    def _fill_small_holes(self, mask: np.ndarray) -> np.ndarray:
        if self.fill_holes_up_to == 0 or not mask.any():
            return mask

        background, _ = ndi.label(~mask, structure=ndi.generate_binary_structure(3, 1))
        boundary = np.concatenate(
            (
                background[0].ravel(),
                background[-1].ravel(),
                background[:, 0].ravel(),
                background[:, -1].ravel(),
                background[:, :, 0].ravel(),
                background[:, :, -1].ravel(),
            )
        )
        exterior = np.unique(boundary)
        counts = np.bincount(background.ravel())
        small = np.flatnonzero(counts * self._voxel_volume <= self.fill_holes_up_to)
        fill_ids = np.setdiff1d(small, np.append(exterior, 0), assume_unique=False)
        if fill_ids.size:
            mask = mask | np.isin(background, fill_ids)
        return mask

    def _filter_components(
        self, mask: np.ndarray, probability: np.ndarray, low: float
    ) -> np.ndarray:
        components, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
        if count == 0:
            return self._empty_labels(mask.shape)

        return self._filter_labeled_regions(components, probability, low)

    def _filter_labeled_regions(
        self, components: np.ndarray, probability: np.ndarray, low: float
    ) -> np.ndarray:
        """Filter and consecutively relabel connected or watershed regions."""

        sizes = np.bincount(components.ravel()) * self._voxel_volume
        valid = np.flatnonzero(sizes >= self.min_particle_volume)
        valid = valid[valid != 0]
        if valid.size == 0:
            return self._empty_labels(components.shape)

        if not self.multiple_particles:
            evidence = np.maximum(probability - low, 0.0)
            scores = ndi.sum(evidence, components, index=valid) * self._voxel_volume
            valid = valid[[int(np.argmax(scores))]]

        kept = np.where(np.isin(components, valid), components, 0)
        _, labels = np.unique(kept, return_inverse=True)
        return labels.reshape(components.shape).astype(np.int32)

    def _split_labels(self, mask: np.ndarray) -> np.ndarray:
        distance = ndi.distance_transform_edt(mask, sampling=self.spacing)
        maxima = (
            morphology.h_maxima(distance, self.watershed_h)
            if self.watershed_h > 0
            else morphology.local_maxima(distance)
        )
        markers, marker_count = ndi.label(maxima, structure=np.ones((3, 3, 3)))
        if marker_count < 2:
            return ndi.label(mask, structure=np.ones((3, 3, 3)))[0].astype(np.int32)
        labels = segmentation.watershed(-distance, markers, mask=mask)
        return labels.astype(np.int32)

    def _make_dem_safe(self, mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """Return the smallest acceptable genus-zero repair, or an empty mask."""
        original_volume = int(mask.sum())
        boundary_touching = any(
            face.any()
            for face in (
                mask[0],
                mask[-1],
                mask[:, 0],
                mask[:, -1],
                mask[:, :, 0],
                mask[:, :, -1],
            )
        )
        if original_volume == 0 or boundary_touching:
            return np.zeros_like(mask), {"dem_safe_rejected": True}

        step = float(self.spacing.min())
        radii = np.arange(0.0, self.max_tunnel_radius + step / 2, step)
        for radius in radii:
            if radius:
                footprint = self._physical_ball(float(radius))
                pad = int(np.ceil(radius / step))
                repaired = ndi.binary_closing(np.pad(mask, pad), structure=footprint)
                repaired = ndi.binary_fill_holes(repaired)
                repaired = repaired[(slice(pad, -pad),) * 3]
            else:
                repaired = ndi.binary_fill_holes(mask)

            changed = float(np.count_nonzero(repaired != mask) / original_volume)
            if changed <= self.max_repair_fraction and self._topology_signature(
                repaired
            ) == (1, 0, 0):
                return repaired, {
                    "dem_safe_rejected": False,
                    "dem_safe_repaired": bool(changed),
                    "dem_safe_repair_method": "morphological_closing",
                    "dem_safe_repair_radius": float(radius),
                    "dem_safe_change_fraction": changed,
                }

        return np.zeros_like(mask), {
            "dem_safe_rejected": True,
            "dem_safe_repaired": False,
        }

    def segment(
        self, probability_grid: np.ndarray, *, return_labels: bool = False
    ) -> np.ndarray:
        """Return a cleaned binary mask or an integer-labeled volume."""
        probability = self._prepare(probability_grid)
        dynamic_range = float(np.ptp(probability))
        if dynamic_range <= 1e-6:
            mask = probability >= 0.5
            labels = self._filter_components(mask, probability, 0.5)
            self.last_thresholds = (0.5, 0.5)
        else:
            if self.smoothing_sigma:
                sigma = self.smoothing_sigma / self.spacing
                filtered = ndi.gaussian_filter(probability, sigma=sigma, mode="reflect")
            else:
                filtered = probability

            low, high = self._select_thresholds(filtered)
            self.last_thresholds = (low, high)
            candidate = filtered >= low
            seeds = filtered >= high

            if not seeds.any() and self.thresholds is None and filtered.max() > low:
                seeds.flat[int(np.argmax(filtered))] = True

            candidate_labels, _ = ndi.label(candidate, structure=np.ones((3, 3, 3)))
            supported = np.unique(candidate_labels[seeds])
            supported = supported[supported != 0]
            mask = np.isin(candidate_labels, supported)

            if self.bridge_radius and mask.any():
                footprint = self._physical_ball(self.bridge_radius)
                relaxed_support = filtered >= max(0.0, low - self.hysteresis_margin)
                mask = ndi.binary_closing(mask, structure=footprint) & relaxed_support

            mask = self._fill_small_holes(mask)
            labels = self._filter_components(mask, filtered, low)

            if self.split_touching and labels.any():
                labels = self._split_labels(labels > 0)
                labels = self._filter_labeled_regions(labels, filtered, low)

        dem_report: dict[str, object] = {}
        if self.dem_safe:
            if labels.any():
                safe_mask, dem_report = self._make_dem_safe(labels > 0)
                labels = safe_mask.astype(np.int32)
            else:
                dem_report = {"dem_safe_rejected": True}

        self.last_labels = labels
        foreground = labels > 0
        component_count = int(labels.max())
        boundary_touching = bool(
            foreground.any()
            and any(
                face.any()
                for face in (
                    foreground[0],
                    foreground[-1],
                    foreground[:, 0],
                    foreground[:, -1],
                    foreground[:, :, 0],
                    foreground[:, :, -1],
                )
            )
        )
        self.last_report = {
            "thresholds": self.last_thresholds,
            "component_count": component_count,
            "foreground_volume": float(foreground.sum() * self._voxel_volume),
            "boundary_touching": boundary_touching,
            **dem_report,
        }
        return labels.copy() if return_labels else foreground

    @staticmethod
    def _topology_signature(mask: np.ndarray) -> tuple[int, int, int]:
        _, components = ndi.label(mask, structure=np.ones((3, 3, 3)))
        background, background_count = ndi.label(
            ~mask, structure=ndi.generate_binary_structure(3, 1)
        )
        exterior = np.unique(
            np.concatenate(
                (
                    background[0].ravel(),
                    background[-1].ravel(),
                    background[:, 0].ravel(),
                    background[:, -1].ravel(),
                    background[:, :, 0].ravel(),
                    background[:, :, -1].ravel(),
                )
            )
        )
        exterior_count = int(np.count_nonzero(exterior != 0))
        cavities = background_count - exterior_count
        cavities = max(0, cavities)
        # For a compact 3D solid, total boundary genus is C + H - chi:
        # foreground components + enclosed cavities - solid Euler characteristic.
        genus = int(components + cavities - measure.euler_number(mask, connectivity=3))
        return int(components), cavities, genus

    def apply(self, probability_grid: np.ndarray) -> tm.Trimesh:
        """Segment ``probability_grid`` and extract a topology-guarded mesh."""
        mask = self.segment(probability_grid)
        if not mask.any():
            return tm.Trimesh(
                vertices=np.empty((0, 3)),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )

        padded = np.pad(mask, 1)
        field = padded.astype(np.float32)
        level = 0.5
        surface_regularized = False
        if self.surface_sigma:
            inside = ndi.distance_transform_edt(padded, sampling=self.spacing)
            outside = ndi.distance_transform_edt(~padded, sampling=self.spacing)
            signed_distance = inside - outside
            regularized = ndi.gaussian_filter(
                signed_distance,
                sigma=self.surface_sigma / self.spacing,
                mode="nearest",
            )
            proposed = regularized >= 0
            if not self.preserve_topology or self._topology_signature(
                proposed
            ) == self._topology_signature(padded):
                field = regularized
                level = 0.0
                surface_regularized = True

        vertices, faces, _, _ = measure.marching_cubes(
            field,
            level=level,
            spacing=tuple(self.spacing),
            allow_degenerate=False,
        )
        vertices -= self.spacing
        mesh = tm.Trimesh(vertices=vertices, faces=faces, process=True)
        tm.repair.fix_normals(mesh)

        if self.autoclean:
            mesh = MeshRepair().apply(mesh)

        self.last_report["surface_regularized"] = surface_regularized
        self.last_report["watertight"] = bool(mesh.is_watertight)
        self.last_report["valid_volume"] = bool(mesh.is_volume)
        genus = mesh_genus(mesh) if mesh.is_watertight else None
        self.last_report["genus"] = genus
        if self.dem_safe:
            mesh_safe = bool(
                mesh.is_watertight
                and mesh.is_volume
                and mesh.body_count == 1
                and genus == 0
            )
            self.last_report["dem_safe_mesh"] = mesh_safe
            if not mesh_safe:
                self.last_report["dem_safe_rejected"] = True
                return tm.Trimesh(
                    vertices=np.empty((0, 3)),
                    faces=np.empty((0, 3), dtype=np.int64),
                    process=False,
                )
        return mesh


class SimpleThreshold(GridProcess):
    """Extract a mesh at a fixed probability threshold.

    Args:
        threshold: Isosurface level to extract.
        autoclean: Repair the extracted mesh with the optional PyMeshFix and
            PyVista dependencies.
    """

    def __init__(self, threshold: float = 0.5, autoclean: bool = False):
        self.threshold = threshold
        self.autoclean = autoclean

    def apply(self, probability_grid: np.ndarray) -> tm.Trimesh:
        """Extract and optionally repair the configured isosurface."""
        return _mesh_from_probability_grid(
            probability_grid, self.threshold, autoclean=self.autoclean
        )
