"""Shape-factor calculations for 2D particle projections."""

import warnings
from os import PathLike
import pathlib
from typing import Optional, Union
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import cv2
import trimesh as tm
from ..mesh_projection import project_onto_axes
from tqdm.auto import tqdm


def compute_sphericity(cnt: np.ndarray) -> float:
    """Compute the sphericity (circularity) of a 2D contour.

    Defined as ``4 * pi * A / P^2`` where *A* is the contour area and *P*
    is its perimeter. A perfect circle yields 1.0.

    *P* is the only tessellation-sensitive term in any of the shape factors: a
    densely triangulated mesh projects a jagged outline whose every wiggle adds
    length, so a decimated mesh of the same particle reads as more spherical.
    That is corrected upstream of this function, by measuring both populations'
    outlines at a common scale -- see :func:`resample_outline` and
    :func:`common_mesh_scale`.

    Args:
        cnt: OpenCV contour array.

    Returns:
        Sphericity value, or ``NaN`` if the perimeter is zero.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return np.nan
    return (4 * np.pi * area) / (perimeter**2)


def compute_convexity(cnt: np.ndarray) -> float:
    """Compute the convexity of a 2D contour.

    Defined as the ratio of the contour area to its convex hull area.
    A perfectly convex shape yields 1.0.

    Args:
        cnt: OpenCV contour array.

    Returns:
        Convexity value, or ``NaN`` if the hull area is zero.
    """
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        convexity = area / hull_area
    else:
        convexity = np.nan

    return convexity


def compute_aspect_ratio(cnt: np.ndarray) -> float:
    """Compute the aspect ratio of a 2D contour.

    Uses the minimum-area oriented bounding rectangle and returns the
    ratio of the shorter side to the longer side. A square yields 1.0.

    Args:
        cnt: OpenCV contour array.

    Returns:
        Aspect ratio in ``(0, 1]``, or ``NaN`` if the shorter side is zero.
    """
    rect = cv2.minAreaRect(cnt)
    (center), (w, h), angle = rect
    oriented_aspect_ratio = min(w, h) / max(w, h) if min(w, h) > 0 else np.nan
    return oriented_aspect_ratio


def _get_contours(image_bin: np.ndarray) -> list[np.ndarray]:
    """Extract external contours from a binary image.

    The image is thresholded at 127 and contours are found using
    ``cv2.RETR_EXTERNAL``.

    Args:
        image_bin: Grayscale or binary image as a NumPy array.

    Returns:
        List of OpenCV contour arrays.
    """
    if image_bin.max() > 0 and image_bin.max() < 255:
        image_bin = (image_bin / image_bin.max() * 255).astype(np.uint8)
    else:
        image_bin = image_bin.astype(np.uint8)

    _, image_bin = cv2.threshold(image_bin, 127, 255, cv2.THRESH_BINARY)

    img_copy = np.ascontiguousarray(image_bin.copy())
    cnt, _ = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnt


@dataclass(slots=True)
class Projection2D:
    """Container for a single 2D projection and its shape metrics.

    Attributes:
        p_id: Unique identifier for the parent particle.
        projection: Image data (binary silhouette) or contour points (outline).
        kind: Projection type -- ``"outline"`` or ``"silhouette"``.
        sphericity: Computed sphericity (circularity).
        aspect_ratio: Computed aspect ratio.
        convexity: Computed convexity.
    """

    p_id: int
    projection: np.ndarray
    kind: str
    sphericity: float = 0.0
    aspect_ratio: float = 0.0
    convexity: float = 0.0

    def calculate_metrics(self) -> None:
        """Compute sphericity, aspect ratio, and convexity from the projection.

        Raises:
            ValueError: If ``kind`` is not ``"outline"`` or ``"silhouette"``.
        """
        if self.kind not in ("outline", "silhouette"):
            raise ValueError("Projection kind must be 'outline' or 'silhouette'.")

        if self.kind == "outline":
            cnt = self.projection.astype(np.float32).reshape((-1, 1, 2))
        elif self.kind == "silhouette":
            cnts = _get_contours(self.projection)
            if len(cnts) == 1:
                cnt = cnts[0]
            elif len(cnts) > 1:
                cnt = max(cnts, key=cv2.contourArea)
                warnings.warn(
                    "Multiple contours found in silhouette projection. Using the largest one."
                )
            else:
                return

        self.sphericity = compute_sphericity(cnt)
        self.aspect_ratio = compute_aspect_ratio(cnt)
        self.convexity = compute_convexity(cnt)


@dataclass(slots=True)
class ShapeFactorResult:
    """Shape factor values for a single particle/projection.

    Attributes:
        sphericity: Sphericity (circularity) value.
        aspect_ratio: Aspect ratio value.
        convexity: Convexity value.
        p_id: Unique identifier for the particle.
    """

    sphericity: float
    aspect_ratio: float
    convexity: float
    p_id: int


# Same as above, but to differentate from
@dataclass(slots=True)
class ShapeFactorBatch:
    """Aggregated shape factor results for multiple particles.

    Supports in-place addition (``+=``) of individual
    :class:`ShapeFactorResult` or other :class:`ShapeFactorBatch`
    instances.

    Attributes:
        sphericities: List of sphericity values.
        aspect_ratios: List of aspect ratio values.
        convexities: List of convexity values.
        p_ids: List of particle identifiers.
    """

    sphericities: list[float]
    aspect_ratios: list[float]
    convexities: list[float]
    p_ids: list[int]

    def __iadd__(self, other: "Union[ShapeFactorResult, ShapeFactorBatch]"):
        """Merge another result into this instance in-place.

        Args:
            other: A single or multi-particle result to append.

        Returns:
            This instance with the merged data.

        Raises:
            ValueError: If *other* is not a supported type.
        """
        if isinstance(other, ShapeFactorResult):
            self.sphericities.append(other.sphericity)
            self.aspect_ratios.append(other.aspect_ratio)
            self.convexities.append(other.convexity)
            self.p_ids.append(other.p_id)
        elif isinstance(other, ShapeFactorBatch):
            self.sphericities.extend(other.sphericities)
            self.aspect_ratios.extend(other.aspect_ratios)
            self.convexities.extend(other.convexities)
            self.p_ids.extend(other.p_ids)
        else:
            raise ValueError("Can only add ShapeFactorResult or ShapeFactorBatch.")
        return self

    def to_array(self):
        """Convert to a stacked NumPy array.

        Returns:
            Array of shape ``(4, N)`` with rows for sphericity, aspect
            ratio, convexity, and particle ID.
        """
        return np.vstack(
            [self.sphericities, self.aspect_ratios, self.convexities, self.p_ids]
        )


def _handle_stl_projection(
    imgs: list[np.ndarray],
    kind: str,
    p_id: int,
    outline_chord: Optional[float] = None,
) -> ShapeFactorBatch:
    """Compute shape factors for multiple projection images of a single STL mesh.

    Args:
        imgs: List of 2D projection images (contours or silhouettes).
        kind: Projection type (``"outline"`` or ``"silhouette"``).
        p_id: Particle identifier.
        outline_chord: Arc-length step to re-sample each outline at, see
            :func:`resample_outline`. ``None`` measures every wiggle.

    Returns:
        Aggregated shape factors across all projections.
    """

    sf_multi = ShapeFactorBatch(
        sphericities=[], aspect_ratios=[], convexities=[], p_ids=[]
    )

    for img in imgs:
        if outline_chord is not None:
            img = resample_outline(img, outline_chord)
            if img is None:
                continue
        proj = Projection2D(projection=img, kind=kind, p_id=p_id)
        proj.calculate_metrics()
        sf_single = ShapeFactorResult(
            sphericity=proj.sphericity,
            aspect_ratio=proj.aspect_ratio,
            convexity=proj.convexity,
            p_id=p_id,
        )
        sf_multi += sf_single

    return sf_multi


def _face_count(source: "tm.Trimesh | PathLike | str") -> int:
    """Face count of a mesh, reading it from the STL header where possible."""
    if isinstance(source, tm.Trimesh):
        return len(source.faces)

    path = pathlib.Path(source)
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) == 84:
        # A binary STL stores its triangle count in bytes 80--84 and is exactly
        # 84 + 50n bytes long. The size check is what makes this safe: some
        # binary files also begin with "solid", so the magic string cannot be
        # trusted on its own.
        count = int(np.frombuffer(header[80:84], dtype="<u4")[0])
        if path.stat().st_size == 84 + 50 * count:
            return count
    return len(tm.load_mesh(str(path)).faces)


def common_mesh_scale(
    reference_meshes: "tm.Trimesh | PathLike | str | list",
) -> tuple[float, float]:
    """Measure the common scale to compare two mesh populations at.

    Pass the **coarser** population -- usually the decimated generations. Its
    triangle size is the finest detail both sets genuinely share, so it sets the
    scale at which the comparison is honest. Use the one returned pair for every
    population in the comparison; deriving it separately per population
    reintroduces exactly the mismatch it exists to remove.

    Returns:
        ``(chord, density)``. *chord* is the median edge length, for
        :func:`resample_outline`. *density* is the median face count per unit
        surface area, for decimating a finer population to match. Both are
        needed: on null controls of one geometry at two tessellations, together
        they cut the sphericity artefact 15x and 28x, where either alone
        manages 5x and 3x.

    Raises:
        ValueError: If no mesh is supplied.
    """
    if isinstance(reference_meshes, (tm.Trimesh, PathLike, str)):
        reference_meshes = [reference_meshes]
    meshes = [
        mesh if isinstance(mesh, tm.Trimesh) else tm.load_mesh(str(mesh))
        for mesh in reference_meshes
    ]
    if not meshes:
        raise ValueError("common_mesh_scale requires at least one mesh")
    chord = float(np.median([np.median(m.edges_unique_length) for m in meshes]))
    density = float(np.median([len(m.faces) / m.area for m in meshes]))
    return chord, density


def resample_outline(
    outline: np.ndarray, chord: float, min_points: int = 12
) -> Optional[np.ndarray]:
    """Re-sample a closed outline at a fixed step along its arc length.

    Walks the boundary in steps of *chord* and keeps the points landed on, so
    every outline is measured with the same ruler no matter how finely its mesh
    was triangulated. Detail shorter than *chord* is stepped over, which is what
    decouples the perimeter -- and therefore sphericity -- from triangle
    density.

    Every returned point lies **on** the original curve, which is what
    distinguishes this from simplifying the contour with
    ``cv2.approxPolyDP``. That drops vertices by perpendicular deviation from a
    chord, and a concave dent *is* a perpendicular deviation, so it flattens
    dents shallower than its tolerance and pins convexity at exactly 1.0. Step
    sampling preserves a dent's depth as long as the dent is wider than
    *chord*.

    Args:
        outline: Closed outline as an ``(n, 2)`` array of points.
        chord: Step length in the outline's own units, from
            :func:`common_mesh_scale`.
        min_points: Floor on the returned point count, so a nearly degenerate
            outline cannot collapse to a triangle and report a garbage
            sphericity.

    Returns:
        The re-sampled outline, or ``None`` if it is too short or degenerate to
        resample.
    """
    xy = np.asarray(outline, dtype=float)
    if len(xy) < 4:
        return None
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack((xy, xy[0]))

    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    )
    if arc[-1] <= 0:
        return None

    count = max(min_points, int(round(arc[-1] / chord)))
    # endpoint=False because the curve is closed: sampling both 0 and arc[-1]
    # would duplicate the start point.
    steps = np.linspace(0.0, arc[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(steps, arc, xy[:, 0]), np.interp(steps, arc, xy[:, 1])]
    )


def _match_face_density(
    mesh: tm.Trimesh, density: float, min_face_count: int = 200
) -> tm.Trimesh:
    """Decimate *mesh* to *density* faces per unit area, if it is finer."""
    target = max(min_face_count, int(round(mesh.area * density)))
    if target >= len(mesh.faces):
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target)


def _process_single_projection(args) -> ShapeFactorResult | ShapeFactorBatch:
    """Worker function for parallel shape factor computation.

    Handles both image files (PNG/JPG silhouettes) and STL meshes
    (projected as outlines). For STL inputs, applies random rotations
    and multi-view projections.

    Args:
        args: Tuple of ``(image_or_path, kind, p_id, rotations_per_stl,
            views_per_rotation, outline_chord, match_face_density, seed)``.

    Returns:
        A :class:`ShapeFactorResult` or :class:`ShapeFactorBatch`.
        Failures are represented by a single result containing NaN values.
    """
    (
        img,
        kind,
        p_id,
        rotations_per_stl,
        views_per_rotation,
        outline_chord,
        match_face_density,
        seed,
    ) = args
    mesh = None
    # A local generator, so every worker sees the same orientations and the
    # process-wide stream is left alone.
    rng = np.random.RandomState(seed)

    # This loop handles file IO type inputs
    if isinstance(img, (PathLike, str)):
        path = pathlib.Path(img)
        if kind == "silhouette" and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            # Black particles on white background
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            cnts = _get_contours(img)
            if len(cnts) == 1:
                img = cnts[0]
            elif len(cnts) > 1:
                img = max(cnts, key=cv2.contourArea)
                warnings.warn(
                    f"Multiple contours found in silhouette projection for {img}. Using the largest one."
                )
            else:
                warnings.warn(
                    f"No contours found in silhouette projection for {img}. Returning NaNs."
                )
                return ShapeFactorResult(np.nan, np.nan, np.nan, p_id)

        elif kind == "outline" and path.suffix.lower() == ".stl":
            mesh = tm.load_mesh(str(path))
    elif kind == "outline" and isinstance(img, tm.Trimesh):
        mesh = img.copy()

    if mesh is not None:
        if match_face_density is not None:
            mesh = _match_face_density(mesh, match_face_density)
        mesh.vertices = mesh.vertices - mesh.center_mass

        def pick_axes():
            if views_per_rotation == 3:
                return ["x", "y", "z"]
            return rng.choice(
                ["x", "y", "z"], size=views_per_rotation, replace=False
            ).tolist()

        if not rotations_per_stl:
            imgs = project_onto_axes(mesh, axes=pick_axes())
            sf = _handle_stl_projection(imgs, kind, p_id, outline_chord)
        else:
            sf = ShapeFactorBatch(
                sphericities=[], aspect_ratios=[], convexities=[], p_ids=[]
            )
            for _ in range(rotations_per_stl):
                # Seeded from `rng`, so both populations in a comparison are
                # viewed from the same orientations rather than from
                # independent draws that add spread of their own.
                mesh.apply_transform(
                    tm.transformations.random_rotation_matrix(rand=rng.rand(3))
                )
                imgs = project_onto_axes(mesh, axes=pick_axes())
                sf += _handle_stl_projection(imgs, kind, p_id, outline_chord)
        return sf
    try:
        # Create fresh object for each process
        proj = Projection2D(projection=img, kind=kind, p_id=p_id)
        proj.calculate_metrics()
        return ShapeFactorResult(
            sphericity=proj.sphericity,
            aspect_ratio=proj.aspect_ratio,
            convexity=proj.convexity,
            p_id=p_id,
        )
    except Exception as exc:
        # Return NaNs on failure to keep lists aligned
        warnings.warn(f"Projection {p_id} failed: {exc}. Returning NaNs.")
        return ShapeFactorResult(np.nan, np.nan, np.nan, p_id)


def compute_shape_factors(
    images: tm.Trimesh | list[np.ndarray | PathLike | tm.Trimesh],
    kind: str,
    num_workers: Optional[int] = None,
    rotations_per_stl: Optional[int] = None,
    views_per_rotation: Optional[int] = None,
    outline_chord: Optional[float] = None,
    match_face_density: Optional[float] = None,
    seed: int = 0,
    progress: bool = True,
) -> ShapeFactorBatch:
    """Compute shape factors for a batch of images or meshes in parallel.

    Args:
        images: A single Trimesh, or a list of Trimeshes, image arrays, image
            file paths, or STL file paths.
        kind: Projection type (``"outline"`` or ``"silhouette"``).
        num_workers: Number of multiprocessing workers. Defaults to the
            number of CPUs.
        rotations_per_stl: Number of random rotations per STL mesh.
        views_per_rotation: Number of axis projections per rotation (1--3).
        outline_chord: Arc-length step to re-sample every outline at, from
            :func:`common_mesh_scale`. Pass the same value to every population
            in a comparison, otherwise sphericity measures triangle density
            rather than shape.
        match_face_density: Faces per unit surface area to decimate each mesh
            down to, from :func:`common_mesh_scale`. Meshes already at or below
            it are left alone. Use it together with *outline_chord*: the pair
            equalises the geometry and the measurement respectively, and on
            null controls each alone leaves several times more artefact than
            the two together.
        seed: Seeds the rotations and the random axis choice. The default makes
            two calls directly comparable; vary it for an independent sample.

    Returns:
        Aggregated shape factor results for all inputs.
    """

    if isinstance(images, tm.Trimesh):
        images = [images]

    if num_workers is None:
        num_workers = mp.cpu_count()

    tasks = [
        (
            img,
            kind,
            p_id,
            rotations_per_stl,
            views_per_rotation,
            outline_chord,
            match_face_density,
            seed,
        )
        for p_id, img in enumerate(images)
    ]

    sf_result = ShapeFactorBatch(
        sphericities=[],
        aspect_ratios=[],
        convexities=[],
        p_ids=[],
    )

    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap(_process_single_projection, tasks)
        for result in tqdm(
            results,
            total=len(tasks),
            desc="Computing shape factors",
            unit="STL" if kind == "outline" else "image",
            disable=not progress,
        ):
            sf_result += result

    return sf_result


def _shape_factors_3d(mesh: tm.Trimesh, p_id: int = 0) -> ShapeFactorResult:
    """Compute 3D shape factors for one validated volume mesh."""
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("Expected a Trimesh")
    if not mesh.is_volume:
        raise ValueError("3D shape factors require a watertight volume mesh")

    volume = float(mesh.volume)
    hull_volume = float(mesh.convex_hull.volume)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    covariance = (np.trace(inertia) * np.eye(3) / 2.0 - inertia) / volume
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    major, minor = np.sqrt(eigenvalues[-1]), np.sqrt(eigenvalues[0])

    return ShapeFactorResult(
        sphericity=(
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / mesh.area
            if mesh.area > 0
            else np.nan
        ),
        aspect_ratio=minor / major if major > 0 else np.nan,
        convexity=volume / hull_volume if hull_volume > 0 else np.nan,
        p_id=p_id,
    )


def _process_single_mesh(args) -> ShapeFactorResult:
    """Compute 3D shape factors for one mesh or STL path."""
    mesh_or_path, p_id = args
    try:
        if isinstance(mesh_or_path, (PathLike, str)):
            path = pathlib.Path(mesh_or_path)
            if path.suffix.lower() != ".stl":
                raise ValueError(f"Expected an STL file, got {path}")
            mesh = tm.load_mesh(path)
        else:
            mesh = mesh_or_path

        return _shape_factors_3d(mesh, p_id)
    except Exception as exc:
        warnings.warn(f"Mesh {p_id} failed: {exc}. Returning NaNs.")
        return ShapeFactorResult(np.nan, np.nan, np.nan, p_id)


def compute_shape_factors_3d(
    meshes: tm.Trimesh | PathLike | str | list[tm.Trimesh | PathLike | str],
    num_workers: Optional[int] = None,
    progress: bool = True,
) -> ShapeFactorBatch:
    """Compute 3D sphericity, convexity, and aspect ratio for meshes.

    Args:
        meshes: A Trimesh, STL path, or list containing either.
        num_workers: Number of multiprocessing workers. Defaults to the
            number of CPUs.
        progress: Whether to display a progress bar.

    Returns:
        One set of shape factors per input, in input order.
    """
    if isinstance(meshes, (tm.Trimesh, PathLike, str)):
        meshes = [meshes]

    tasks = [(mesh, p_id) for p_id, mesh in enumerate(meshes)]
    result = ShapeFactorBatch([], [], [], [])

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        results = pool.imap(_process_single_mesh, tasks)
        for item in tqdm(
            results,
            total=len(tasks),
            desc="Computing 3D shape factors",
            unit="mesh",
            disable=not progress,
        ):
            result += item

    return result
