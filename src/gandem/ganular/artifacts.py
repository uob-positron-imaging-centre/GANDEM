"""Save GANular plots, voxel arrays, and STL meshes."""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from stl import mesh


def save_voxel_plot(voxels: torch.Tensor, save_path: str, save_dir: str = "") -> None:
    """Render the first voxel grid in a batch as a 3D matplotlib plot and save to disk.

    The volume is binarised at a 0.5 threshold before visualisation.

    Args:
        voxels: Voxel tensor of shape ``(N, 1, D, H, W)``, ``(N, D, H, W)``,
            or ``(D, H, W)``. Only the first item in the batch is plotted.
        save_path: File path for the output PNG image.
        save_dir: Directory to create if it does not already exist.
    """
    # voxels: (N, 1, D, H, W) or (D, H, W)
    # We visualize the first item in batch
    if len(voxels.shape) == 5:
        voxels = voxels[0, 0]
    elif len(voxels.shape) == 4:
        voxels = voxels[0]

    # voxels is now (D, H, W) with values in [0, 1]
    # Binarize for visualization
    voxels = voxels.detach().cpu().numpy() > 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxels, edgecolor="k")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path)
    plt.close(fig)


def save_loss_plot(d_losses, g_losses, save_path) -> None:
    """Plot discriminator and generator losses over training iterations.

    Args:
        d_losses: Sequence of discriminator loss values.
        g_losses: Sequence of generator loss values.
        save_path: File path for the output PNG image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_projection_plot(projections: torch.Tensor, save_path: str):
    """Save a grid of up to four 2D projection images to disk.

    Args:
        projections: Batch of projection images with shape ``(B, C, H, W)``.
        save_path: File path for the output PNG image.
    """
    # projections: (B, C, H, W) - We visualize the first few in the batch
    projections = projections.detach().cpu()

    # Normalize to [0, 1] for plotting if they aren't already
    if projections.min() < 0:
        projections = (projections + 1) / 2

    # Create a grid of images (e.g., first 4 images)
    num_imgs = min(projections.size(0), 4)
    fig, axes = plt.subplots(1, num_imgs, figsize=(15, 4))

    if num_imgs == 1:
        axes = [axes]

    for i in range(num_imgs):
        img = projections[i].permute(1, 2, 0).squeeze()  # (H, W) or (H, W, C)
        # FIX: Explicit scale to ensure 0 is black and 1 is white
        axes[i].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i].axis("off")
        axes[i].set_title(f"Proj {i + 1}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_voxel_stl(voxels, save_path, threshold=0.5) -> None:
    """Convert a voxel grid to a triangle mesh and save as binary STL.

    Uses the Marching Cubes algorithm to extract an isosurface at the given
    threshold and writes the result via *numpy-stl*.

    Args:
        voxels: Input volume as a ``torch.Tensor`` or ``np.ndarray``.
            Batch/channel dimensions are squeezed automatically.
        save_path: Output file path for the ``.stl`` file.
        threshold: Isosurface level for Marching Cubes.
    """
    if isinstance(voxels, torch.Tensor):
        vol = voxels.detach().cpu().numpy()
    else:
        vol = voxels

    # Remove batch/channel dims to get (D, H, W)
    vol = vol.squeeze()

    if vol.ndim != 3:
        print(f"Expected 3D volume, got shape {vol.shape}")
        return

    # Check if volume contains the threshold level
    if vol.min() > threshold or vol.max() < threshold:
        # print(f"Volume values [{vol.min():.2f}, {vol.max():.2f}] do not cross threshold {threshold}. Skipping STL.")
        return

    try:
        # Marching Cubes to get vertices and faces
        verts, faces, _, _ = measure.marching_cubes(vol, level=threshold)

        # faces.shape[0] is the number of triangles
        obj_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        # Set vectors (numpy-stl expects vertices for each face)
        # verts[faces] creates an array of shape (N_faces, 3, 3)
        obj_mesh.vectors = verts[faces]

        # Save to binary STL
        obj_mesh.save(save_path)

    except Exception as e:
        print(f"Failed to generate STL: {e}")


def save_voxel_npy(voxels: torch.Tensor | np.ndarray, save_path) -> None:
    """Save a voxel grid as a NumPy ``.npy`` file.

    Args:
        voxels: Input volume as a ``torch.Tensor`` or ``np.ndarray``.
        save_path: Output file path for the ``.npy`` file.
    """
    if isinstance(voxels, torch.Tensor):
        vol = voxels.detach().cpu().numpy()
    else:
        vol = voxels

    np.save(save_path, vol)
