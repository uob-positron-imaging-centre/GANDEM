"""
Conversion of STL mesh files to 2D projection images for training GANular.
This module provides functions to batch-convert STL files to 2D projection images in parallel. Each STL is projected along three axes and saved as PNG. When *img_size* is ``None`` it is automatically computed from the largest mesh bounding box, scaled by *image_pad*.
"""

import pathlib
from os import PathLike
from typing import Iterable
import multiprocessing as mp
from functools import partial
import numpy as np
import trimesh as tm
from tqdm.auto import tqdm
import cv2
from ..mesh_projection import project_onto_axes


def _get_mesh_max_bound(file: pathlib.Path) -> float:
    """Compute the maximum bounding-box dimension of a mesh file.

    Args:
        file: Path to a mesh file (STL, OBJ, etc.).

    Returns:
        Maximum extent across all axes, or ``NaN`` on failure.
    """
    try:
        mesh = tm.load(file)
        # Handle cases where bounds might be empty or None
        if hasattr(mesh, "bounds") and mesh.bounds is not None:
            return np.ptp(mesh.bounds, axis=0).max()
        else:
            return np.nan
    except Exception:
        return np.nan


def compute_max_grid_size(
    files: Iterable[pathlib.Path], num_workers: int = None
) -> int:
    """Compute the maximum grid size needed to accommodate all mesh projections.

    Loads each mesh in parallel and returns the largest bounding-box
    dimension across the entire set.

    Args:
        files: Iterable of paths to mesh files.
        num_workers: Number of multiprocessing workers. Defaults to the
            number of CPUs.

    Returns:
        Maximum bounding-box extent (rounded to int).
    """
    if not isinstance(files, list):
        files = list(files)

    if not num_workers:
        num_workers = mp.cpu_count()

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_get_mesh_max_bound, files),
                total=len(files),
                desc="Computing max grid size",
            )
        )

    if not results:
        return 0

    return max(results)


def _single_stl2proj(
    stl_file: PathLike, output_dir: pathlib.Path, img_size: int, auto_scale: bool
) -> None:
    """Convert a single STL mesh to 2D projection PNG images.

    Generates one projection per axis (x, y, z), centres and optionally
    scales each to fit the canvas, and saves them as binary PNGs.

    Args:
        stl_file: Path to the input STL file.
        output_dir: Directory for the output PNG files.
        img_size: Side length of the square output images in pixels.
        auto_scale: Whether to scale each projection to fill 90 % of the
            canvas.
    """

    if isinstance(stl_file, str):
        stl_file = pathlib.Path(stl_file)

    if not stl_file.exists():
        print(f"File not found: {stl_file}")
        return

    try:
        mesh = tm.load(stl_file)
        projections = project_onto_axes(mesh)

        for proj_index, proj in enumerate(projections):
            mask = np.zeros((img_size, img_size), dtype=np.uint8)

            # Calculate bounds
            min_x, max_x = proj[:, 0].min(), proj[:, 0].max()
            min_y, max_y = proj[:, 1].min(), proj[:, 1].max()

            width = max_x - min_x
            height = max_y - min_y
            max_dim = max(width, height)

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            if auto_scale:
                scale = (img_size * 0.9) / max_dim if max_dim > 0 else 1.0

            else:
                scale = 1.0

            centered_proj = (proj - [center_x, center_y]) * scale + [
                img_size / 2,
                img_size / 2,
            ]

            final_img = cv2.fillPoly(mask, [centered_proj.astype(np.int32)], 255)

            final_img = cv2.bitwise_not(final_img)

            output_path = output_dir / f"{stl_file.stem}_{proj_index}.png"

            print(
                f"Saving projection {proj_index} for {stl_file.name} to {output_path}"
            )
            cv2.imwrite(str(output_path), final_img)

    except Exception as e:
        print(f"Error processing {stl_file}: {e}")


def convert_stls_to_projections(
    stl_files: Iterable[PathLike],
    output_dir: PathLike,
    img_size: int = None,
    num_workers: int = None,
    image_pad: float = 1.2,
    auto_scale: bool = True,
):
    """Batch-convert STL files to 2D projection images in parallel.

    Each STL is projected along three axes and saved as PNG. When
    *img_size* is ``None`` it is automatically computed from the largest
    mesh bounding box, scaled by *image_pad*.

    Args:
        stl_files: Iterable of paths to STL mesh files.
        output_dir: Directory for the output PNG files (created if absent).
        img_size: Fixed canvas size in pixels. Auto-computed when ``None``.
        num_workers: Number of multiprocessing workers. Defaults to the
            number of CPUs.
        image_pad: Multiplicative padding factor applied when auto-computing
            *img_size*.
        auto_scale: Whether to scale each projection to fill 90 % of the
            canvas.
    """

    if not isinstance(stl_files, list):
        stl_files = list(stl_files)

    if img_size is None:
        img_size = int(compute_max_grid_size(stl_files, num_workers) * image_pad)
    else:
        # Safety cast in case user passed a float
        img_size = int(img_size)

    if isinstance(output_dir, str):
        output_dir = pathlib.Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=num_workers) as pool:
        process_stl = partial(
            _single_stl2proj,
            output_dir=output_dir,
            img_size=img_size,
            auto_scale=auto_scale,
        )
        list(
            tqdm(
                pool.imap_unordered(process_stl, stl_files),
                total=len(stl_files),
                desc="Processing STL files",
            )
        )
