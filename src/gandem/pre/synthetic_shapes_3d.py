import os
from glob import glob
from os import PathLike
import warnings
import numpy as np
import trimesh as tm
from tqdm.auto import tqdm, trange
from stl import mesh
import cv2
from ..mesh_projection import project_onto_axes


def generate_diamonds3d(
    n: int,
    output_dir: str,
    base_scale: float = 10.0,
    random_skew: bool = False,
):
    """
    Generates 3D STL files of diamond shapes (octahedrons).

    Each diamond has random width, height, and depth, is randomly rotated
    in 3D space, and optionally subjected to a random geometric shear (skew).

    Args:
        output_dir: Directory to save the generated STL files.
        n: Number of files to generate.
        base_scale: Base unit size for the diamond (e.g., mm).
        random_skew: Whether to apply a random 3D shear transformation to
            the mesh vertices.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(n, desc="Generating STLs"):
        h = np.random.uniform(0.8, 1.8) * base_scale  # Height (Y axis)
        w = np.random.uniform(0.5, 1.2) * base_scale  # Width (X axis)
        d = np.random.uniform(0.5, 1.2) * base_scale  # Depth (Z axis)

        vertices = np.array(
            [
                [0, h, 0],  # Top
                [0, -h, 0],  # Bottom
                [0, 0, d],  # Front
                [w, 0, 0],  # Right
                [0, 0, -d],  # Back
                [-w, 0, 0],  # Left
            ]
        )

        rx = np.random.uniform(0, 2 * np.pi)
        ry = np.random.uniform(0, 2 * np.pi)
        rz = np.random.uniform(0, 2 * np.pi)

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        transformation = Rz @ Ry @ Rx

        if random_skew:
            # A shear matrix skews one axis based on the value of another
            shear_xy = np.random.uniform(-0.5, 0.5)  # Shear X based on Y
            shear_xz = np.random.uniform(-0.5, 0.5)  # Shear X based on Z
            shear_yz = np.random.uniform(-0.5, 0.5)  # Shear Y based on Z

            Shear = np.array([[1, shear_xy, shear_xz], [0, 1, shear_yz], [0, 0, 1]])
            transformation = Shear @ transformation

        vertices = vertices @ transformation.T

        faces = np.array(
            [
                [0, 2, 3],  # Top-Front-Right
                [0, 3, 4],  # Top-Right-Back
                [0, 4, 5],  # Top-Back-Left
                [0, 5, 2],  # Top-Left-Front
                [1, 3, 2],  # Bot-Right-Front
                [1, 4, 3],  # Bot-Back-Right
                [1, 5, 4],  # Bot-Left-Back
                [1, 2, 5],  # Bot-Front-Left
            ]
        )

        diamond_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        for j, face in enumerate(faces):
            for k in range(3):
                diamond_mesh.vectors[j][k] = vertices[face[k], :]

        filename = os.path.join(output_dir, f"diamond_{i:04d}.stl")
        diamond_mesh.save(filename)

    print(f"Successfully generated {n} STL files in '{output_dir}'.")


def _get_global_scale(stl_files, img_size, padding_percent=0.2):
    """
    Scans all STLs to find the single largest dimension.
    Returns a scale factor so the largest object fits in the image.
    """
    print("Pass 1: Scanning dataset for global max dimensions...")
    max_dim = 0.0

    for f in tqdm(stl_files, desc="Scanning sizes"):
        try:
            # We use 'primitive' loading if possible for speed, else full load
            mesh = tm.load(f)

            extents = mesh.extents
            curr_max = np.max(extents)

            if curr_max > max_dim:
                max_dim = curr_max

        except Exception as e:
            print(f"Skipping {f} during scan: {e}")

    if max_dim == 0:
        return 1.0

    # Margin for rotation (a cube's diagonal is wider than its side)
    # Approx sqrt(3) be safe for 3D rotation
    max_dim *= 1.75

    target_dim = img_size * (1.0 - padding_percent)
    global_scale = target_dim / max_dim

    return global_scale


def generate_projections(
    n: int,
    stl_dir: PathLike,
    output_dir: PathLike,
    img_size: int = 256,
    filename_prefix: str = "proj",
) -> None:
    """
    Randomly samples STL files from a directory and generates 2D projections.

    Args:
        n: Number of random samples to generate.
        stl_dir: Directory containing STL files to sample from.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stl_files = glob(os.path.join(stl_dir, "*.stl"))
    if not stl_files:
        raise FileNotFoundError(
            f"No STL files found in '{stl_dir}'. Please generate them first."
        )

    global_scale = _get_global_scale(stl_files, img_size)

    print(f"Found {len(stl_files)} STL files. Generating projections...")

    for i in trange(n, desc="Generating Projections"):
        rand_f = np.random.choice(stl_files)
        rand_ax = np.random.choice(["x", "y", "z"])
        try:
            mesh = tm.load(rand_f)

        except Exception as e:
            warnings.warn(f"Error loading '{rand_f}': {e}. Skipping.")
            continue

        rot_matrix = tm.transformations.random_rotation_matrix()
        mesh.apply_transform(rot_matrix)

        projections = project_onto_axes(mesh, axes=[rand_ax])
        coords = projections[0]

        canvas = np.zeros((img_size, img_size), dtype=np.uint8)
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        center_shape = (min_xy + max_xy) / 2

        centered_shape = coords - center_shape
        scaled_shape = centered_shape * global_scale

        center_img = np.array([img_size / 2, img_size / 2])
        poly = np.round(scaled_shape + center_img).astype(np.int32)

        cv2.fillPoly(canvas, [poly], 255)
        img = cv2.bitwise_not(canvas)

        filename = os.path.join(output_dir, f"{filename_prefix}_{i:04d}.png")
        cv2.imwrite(filename, img)
