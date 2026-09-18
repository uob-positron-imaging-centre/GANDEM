"""Advisory quality checks for raw GANular outputs.

The module deliberately measures voxel masks and projections without mesh
repair, smoothing, component removal, or adaptive thresholding.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from skimage import measure
import trimesh as tm

FEATURES_2D = ("circularity", "aspect_ratio", "convexity")
FEATURES_3D = ("convexity_3d", "sphericity", "aspect_ratio_3d")


def mesh_genus(mesh: tm.Trimesh) -> int:
    """Return the total genus of a closed, orientable triangle mesh."""
    if not isinstance(mesh, tm.Trimesh) or not mesh.is_watertight:
        raise ValueError("Genus requires a watertight Trimesh")

    genus = float(mesh.body_count - mesh.euler_number / 2)
    rounded = int(round(genus))
    if rounded < 0 or not np.isclose(genus, rounded):
        raise ValueError("Mesh does not define an integer genus")
    return rounded


@dataclass(slots=True)
class QualityChecks:
    """Configuration for the five advisory model-quality checks."""

    enabled: bool = True
    uct_data_dir: str | None = None
    projection_reference_dir: str | None = None
    evaluation_interval: int = 10
    sample_size: int = 32
    projection_sample_size: int = 32
    evaluation_seed: int = 0
    voxel_threshold: float = 0.5
    projection_threshold: float = 0.5
    distance_ratio_max: float = 3.0
    fidelity_min: float = 0.8
    coverage_min: float = 0.8
    valid_sample_min: float = 0.8
    topology_valid_min: float = 1.0
    saturation_max: float = 0.95
    warmup_steps: int = 100
    gradient_norm_min: float = 1e-10
    gradient_norm_max: float = 1e4

    def validate(self) -> None:
        if (
            self.evaluation_interval < 1
            or self.sample_size < 4
            or self.projection_sample_size < 4
        ):
            raise ValueError(
                "Quality evaluation needs evaluation_interval >= 1, sample_size >= 4, "
                "and projection_sample_size >= 4"
            )
        for name in ("voxel_threshold", "projection_threshold"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "fidelity_min",
            "coverage_min",
            "valid_sample_min",
            "topology_valid_min",
            "saturation_max",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)


def _unit_interval(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size and np.nanmax(array) > 1.0:
        array = array / np.nanmax(array)
    return array


def shape_metrics_2d(
    image: np.ndarray, threshold: float = 0.5, convexity_basis="area"
) -> dict[str, float]:
    """Return circularity, aspect ratio and convexity for one silhouette."""

    if convexity_basis not in ("area", "perimeter"):
        raise ValueError(
            "Basis of shape metric calulation must be 'area' or 'perimeter' "
        )

    mask = (_unit_interval(np.squeeze(image)) > threshold).astype(np.uint8) * 255
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D projection, got shape {mask.shape}")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {name: float("nan") for name in FEATURES_2D}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    _, (width, height), _ = cv2.minAreaRect(contour)
    major, minor = max(width, height), min(width, height)
    hull = cv2.convexHull(contour)

    if convexity_basis == "perimeter":
        hull_perimeter = float(cv2.arcLength(hull, True))
        convexity = hull_perimeter / perimeter if perimeter > 0 else float("nan")
    else:
        hull_area = float(cv2.contourArea(hull))
        convexity = area / hull_area if hull_area > 0 else float("nan")

    return {
        "circularity": (
            4.0 * np.pi * area / perimeter**2 if perimeter > 0 else float("nan")
        ),
        "aspect_ratio": minor / major if major > 0 else float("nan"),
        "convexity": convexity if convexity > 0 else float("nan"),
    }


def shape_metrics_3d(volume: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return 3D convexity, sphericity, and PCA aspect ratio for one volume.

    Convexity is ``volume / convex_hull_volume``.
    Sphericity is ``pi^(1/3) (6 V)^(2/3) / surface_area``.
    """
    mask = _unit_interval(np.squeeze(volume)) > threshold
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {mask.shape}")
    coordinates = np.argwhere(mask)
    if len(coordinates) < 4:
        return {name: float("nan") for name in FEATURES_3D}

    vertices, faces, _, _ = measure.marching_cubes(
        np.pad(mask.astype(np.float32), 1), level=0.5
    )
    mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
    particle_volume = abs(float(mesh.volume))
    convex_hull_volume = float(mesh.convex_hull.volume)

    eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False, bias=True))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    major = float(np.sqrt(eigenvalues[-1]))
    minor = float(np.sqrt(eigenvalues[0]))

    return {
        "convexity_3d": (
            particle_volume / convex_hull_volume
            if convex_hull_volume > 0
            else float("nan")
        ),
        "sphericity": (
            np.pi ** (1.0 / 3.0) * (6.0 * particle_volume) ** (2.0 / 3.0) / mesh.area
            if mesh.area > 0
            else float("nan")
        ),
        "aspect_ratio_3d": minor / major if major > 0 else float("nan"),
    }


def _shape_metrics_mesh(mesh: tm.Trimesh, _threshold: float = 0.5) -> dict[str, float]:
    """Return the volume-equivalent 3D metrics for one STL mesh."""
    if not isinstance(mesh, tm.Trimesh) or not mesh.is_volume:
        return {name: float("nan") for name in FEATURES_3D}

    volume = float(mesh.volume)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    covariance = (np.trace(inertia) * np.eye(3) / 2.0 - inertia) / volume
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    major = float(np.sqrt(eigenvalues[-1]))
    minor = float(np.sqrt(eigenvalues[0]))
    convex_hull_volume = float(mesh.convex_hull.volume)

    return {
        "convexity_3d": (
            volume / convex_hull_volume if convex_hull_volume > 0 else float("nan")
        ),
        "sphericity": (
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / mesh.area
            if mesh.area > 0
            else float("nan")
        ),
        "aspect_ratio_3d": minor / major if major > 0 else float("nan"),
    }


def topology_metrics_3d(
    volume: np.ndarray, threshold: float = 0.5
) -> dict[str, bool | int | None]:
    """Measure raw voxel topology without repairing or removing components."""
    mask = _unit_interval(np.squeeze(volume)) > threshold
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {mask.shape}")

    empty = not bool(mask.any())
    component_count = (
        0 if empty else int(label(mask, structure=generate_binary_structure(3, 1))[1])
    )
    boundary_touching = bool(
        not empty
        and any(
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
    )

    watertight = winding_consistent = valid_volume = False
    genus = None
    if not empty:
        vertices, faces, _, _ = measure.marching_cubes(
            np.pad(mask.astype(np.float32), 1),
            level=0.5,
            gradient_direction="ascent",
            allow_degenerate=False,
        )
        mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
        watertight = bool(mesh.is_watertight)
        winding_consistent = bool(mesh.is_winding_consistent)
        valid_volume = bool(mesh.is_volume)
        if watertight:
            genus = mesh_genus(mesh)

    return {
        "empty": empty,
        "boundary_touching": boundary_touching,
        "component_count": component_count,
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "valid_volume": valid_volume,
        "genus": genus,
        "valid": bool(
            not empty
            and not boundary_touching
            and component_count == 1
            and watertight
            and winding_consistent
            and valid_volume
            and genus == 0
        ),
    }


def check_topology(generated_3d, config: QualityChecks) -> dict:
    """Require raw generated particles to form one contained, valid volume."""
    metrics = [
        topology_metrics_3d(volume, config.voxel_threshold) for volume in generated_3d
    ]
    count = len(metrics)

    def fraction(predicate) -> float:
        return float(np.mean([predicate(item) for item in metrics])) if count else 0.0

    valid_fraction = fraction(lambda item: item["valid"])
    passed = valid_fraction >= config.topology_valid_min
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "threshold": config.voxel_threshold,
        "required_fraction": config.topology_valid_min,
        "samples": count,
        "valid_samples": sum(bool(item["valid"]) for item in metrics),
        "valid_fraction": valid_fraction,
        "empty_fraction": fraction(lambda item: item["empty"]),
        "boundary_touching_fraction": fraction(lambda item: item["boundary_touching"]),
        "single_component_fraction": fraction(
            lambda item: item["component_count"] == 1
        ),
        "watertight_fraction": fraction(lambda item: item["watertight"]),
        "winding_consistent_fraction": fraction(
            lambda item: item["winding_consistent"]
        ),
        "valid_volume_fraction": fraction(lambda item: item["valid_volume"]),
        "genus_zero_fraction": fraction(lambda item: item["genus"] == 0),
    }


def feature_table(
    items, metric_function, threshold: float, names: tuple[str, ...]
) -> np.ndarray:
    """Calculate a finite feature matrix from images or volumes."""
    rows = [
        [metric_function(item, threshold)[name] for name in names] for item in items
    ]
    if not rows:
        return np.empty((0, len(names)), dtype=float)
    table = np.asarray(rows, dtype=float)
    return table[np.isfinite(table).all(axis=1)]


def _standardise(
    real: np.ndarray, generated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(real, axis=0)
    scale = np.percentile(real, 75, axis=0) - np.percentile(real, 25, axis=0)
    scale = np.where(scale > 1e-8, scale, np.std(real, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (real - median) / scale, (generated - median) / scale


def compare_feature_sets(
    real: np.ndarray, generated: np.ndarray, seed: int = 0
) -> dict:
    """Compare fidelity and coverage in a small, standardised feature space."""
    if len(real) < 4 or len(generated) < 4:
        return {
            "status": "skipped",
            "reason": "at least four valid samples are required",
        }

    real_z, generated_z = _standardise(real, generated)
    generated_distance = float(
        np.mean(
            [
                wasserstein_distance(real_z[:, i], generated_z[:, i])
                for i in range(real.shape[1])
            ]
        )
    )

    rng = np.random.default_rng(seed)
    baseline = []
    for _ in range(20):
        shuffled = rng.permutation(len(real_z))
        left, right = np.array_split(real_z[shuffled], 2)
        baseline.append(
            np.mean(
                [
                    wasserstein_distance(left[:, i], right[:, i])
                    for i in range(real.shape[1])
                ]
            )
        )
    real_real_distance = float(np.median(baseline))

    real_distances = distance.cdist(real_z, real_z)
    np.fill_diagonal(real_distances, np.inf)
    k = min(3, len(real_z) - 1)
    support_radius = float(
        np.percentile(np.partition(real_distances, k - 1, axis=1)[:, k - 1], 95)
    )
    cross_distances = distance.cdist(generated_z, real_z)
    fidelity = float(np.mean(cross_distances.min(axis=1) <= support_radius))
    coverage = float(np.mean(cross_distances.min(axis=0) <= support_radius))

    return {
        "status": "ok",
        "generated_real_distance": generated_distance,
        "real_real_distance": real_real_distance,
        "distance_ratio": generated_distance / max(real_real_distance, 0.05),
        "support_radius": support_radius,
        "fidelity": fidelity,
        "coverage": coverage,
    }


def _feature_summary(names, real: np.ndarray, generated: np.ndarray) -> dict:
    return {
        group: {
            name: {"mean": float(values[:, i].mean()), "std": float(values[:, i].std())}
            for i, name in enumerate(names)
        }
        for group, values in (("real", real), ("generated", generated))
        if len(values)
    }


def check_training_health(stats: dict, step: int, config: QualityChecks) -> dict:
    """Evaluate numerical health; warm-up failures are reported but not failed."""
    numeric = [float(value) for value in stats.values()]
    finite = bool(np.isfinite(numeric).all())
    gradients_ok = all(
        config.gradient_norm_min <= float(stats[name]) <= config.gradient_norm_max
        for name in ("generator_gradient_norm", "critic_gradient_norm")
    )
    saturation_ok = float(stats["projection_saturation"]) <= config.saturation_max
    passed = finite and gradients_ok and saturation_ok
    status = (
        "warming_up" if step < config.warmup_steps else ("pass" if passed else "fail")
    )
    return {
        "status": status,
        "passed": passed if step >= config.warmup_steps else None,
        **stats,
    }


def assess_generated_samples(
    real_2d: np.ndarray,
    generated_2d,
    generated_3d,
    config: QualityChecks,
    real_3d: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Evaluate projection agreement, topology, 3D fidelity, and 3D coverage."""
    generated_2d_features = feature_table(
        generated_2d, shape_metrics_2d, config.projection_threshold, FEATURES_2D
    )
    projection = compare_feature_sets(real_2d, generated_2d_features, seed)
    valid_2d_fraction = len(generated_2d_features) / max(len(generated_2d), 1)
    if projection["status"] == "ok":
        projection["passed"] = bool(
            projection["distance_ratio"] <= config.distance_ratio_max
            and projection["fidelity"] >= config.fidelity_min
            and valid_2d_fraction >= config.valid_sample_min
        )
    elif len(real_2d) >= 4 and valid_2d_fraction < config.valid_sample_min:
        projection.update(
            status="fail",
            passed=False,
            reason="too few generated projections contained a measurable particle",
        )
    projection["features"] = FEATURES_2D
    projection["feature_summary"] = _feature_summary(
        FEATURES_2D, real_2d, generated_2d_features
    )
    projection["valid_samples"] = {
        "real": len(real_2d),
        "generated": len(generated_2d_features),
        "generated_fraction": valid_2d_fraction,
    }
    topology = check_topology(generated_3d, config)

    if real_3d is None:
        skipped = {"status": "skipped", "reason": "uct_data_dir was not provided"}
        return {
            "projection_quality": projection,
            "topology": topology,
            "volume_fidelity": skipped,
            "volume_coverage": skipped,
        }

    generated_3d_features = feature_table(
        generated_3d, shape_metrics_3d, config.voxel_threshold, FEATURES_3D
    )
    comparison = compare_feature_sets(real_3d, generated_3d_features, seed)
    valid_3d_fraction = len(generated_3d_features) / max(len(generated_3d), 1)
    fidelity = {**comparison, "features": FEATURES_3D}
    coverage = {**comparison, "features": FEATURES_3D}
    feature_summary = _feature_summary(FEATURES_3D, real_3d, generated_3d_features)
    valid_samples = {
        "real": len(real_3d),
        "generated": len(generated_3d_features),
        "generated_fraction": valid_3d_fraction,
    }
    fidelity["feature_summary"] = feature_summary
    coverage["feature_summary"] = feature_summary
    fidelity["valid_samples"] = valid_samples
    coverage["valid_samples"] = valid_samples
    if comparison["status"] == "ok":
        fidelity["passed"] = bool(
            comparison["distance_ratio"] <= config.distance_ratio_max
            and comparison["fidelity"] >= config.fidelity_min
            and valid_3d_fraction >= config.valid_sample_min
        )
        coverage["passed"] = bool(
            comparison["coverage"] >= config.coverage_min
            and valid_3d_fraction >= config.valid_sample_min
        )
    elif len(real_3d) >= 4 and valid_3d_fraction < config.valid_sample_min:
        for check in (fidelity, coverage):
            check.update(
                status="fail",
                passed=False,
                reason="too few generated volumes contained a measurable particle",
            )
    return {
        "projection_quality": projection,
        "topology": topology,
        "volume_fidelity": fidelity,
        "volume_coverage": coverage,
    }


def load_uct_features(config: QualityChecks) -> np.ndarray | None:
    """Load up to ``sample_size`` raw STL uCT meshes and calculate 3D features."""
    if not config.uct_data_dir:
        return None
    paths = sorted(
        path
        for path in Path(config.uct_data_dir).glob("*")
        if path.suffix.lower() == ".stl"
    )
    if not paths:
        raise ValueError(f"No STL uCT meshes found in {config.uct_data_dir}")

    rng = np.random.default_rng(config.evaluation_seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[: config.sample_size]]
    meshes = [tm.load_mesh(path) for path in paths]
    return feature_table(
        meshes, _shape_metrics_mesh, config.voxel_threshold, FEATURES_3D
    )


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one JSON record to a diagnostics file."""

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean(record), allow_nan=False) + "\n")
